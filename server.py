"""MLB scoreboard server.

Offloads the ESP32's data work: one GET /teamInfo/{teamId} returns a pre-derived
JSON object that maps 1:1 onto the DisplayData class in data.h. All MLB StatsAPI
fetching, parsing, and derivation that net.cpp did on-device happens here, behind
a per-endpoint TTL cache so many device polls collapse into a few upstream calls.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlb-server")

app = FastAPI(title="MLB Scoreboard Server")

# --- Upstream fetch + per-URL TTL cache -------------------------------------

BASE = "https://statsapi.mlb.com"

# Shared HTTP/2-capable client, opened/closed with the app lifecycle.
_client: httpx.AsyncClient | None = None

# url -> (expires_at_epoch, payload)
_cache: dict[str, tuple[float, dict]] = {}
# url -> lock, so concurrent misses on the same url fetch once (no thundering herd)
_locks: dict[str, asyncio.Lock] = {}


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=15.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


async def fetch_json(url: str, ttl: float, not_found_is_404: bool = False) -> dict:
    """GET `url`, caching the parsed body for `ttl` seconds keyed by full URL.

    Raises HTTPException(502) on any upstream/transport/parse failure. If
    `not_found_is_404` is set, an upstream 404 is surfaced as HTTPException(404)
    instead — MLB returns 404 for an unknown teamId.
    """
    now = time.monotonic()
    hit = _cache.get(url)
    if hit is not None and hit[0] > now:
        return hit[1]

    lock = _locks.setdefault(url, asyncio.Lock())
    async with lock:
        # Another coroutine may have populated the cache while we waited.
        hit = _cache.get(url)
        now = time.monotonic()
        if hit is not None and hit[0] > now:
            return hit[1]

        log.info("cache miss -> %s", url)
        try:
            resp = await _client.get(url)  # type: ignore[union-attr]
            if not_found_is_404 and resp.status_code == 404:
                raise HTTPException(status_code=404, detail="unknown teamId")
            resp.raise_for_status()
            payload = resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            log.warning("upstream fetch failed for %s: %s", url, exc)
            raise HTTPException(status_code=502, detail="upstream MLB API error") from exc

        # Drop expired keys on write; schedule/live-feed URLs vary by date/gamePk
        # and would otherwise accumulate for the life of the process.
        now = time.monotonic()
        for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
            del _cache[k]

        _cache[url] = (now + ttl, payload)
        return payload


# --- Pydantic models mirroring data.h ---------------------------------------


class TeamInfo(BaseModel):
    name: str = ""
    abbrev: str = ""
    score: int = 0
    odds: str = ""  # no source in the live feed (see net.cpp:279); kept for 1:1 mapping


class BatterInfo(BaseModel):
    name: str = ""
    hits: int = 0
    atBats: int = 0


class PitcherInfo(BaseModel):
    name: str = ""
    pitchCount: int = 0


class BaseRunners(BaseModel):
    first: bool = False
    second: bool = False
    third: bool = False


class CurrentGame(BaseModel):
    gameId: int = 0
    home: TeamInfo = TeamInfo()
    away: TeamInfo = TeamInfo()
    batter: BatterInfo = BatterInfo()
    pitcher: PitcherInfo = PitcherInfo()
    isTopInning: bool = True  # replaces data.h InningHalf enum
    inning: int = 1
    outs: int = 0
    strikes: int = 0
    balls: int = 0
    runners: BaseRunners = BaseRunners()


class NextGame(BaseModel):
    home: TeamInfo = TeamInfo()
    away: TeamInfo = TeamInfo()
    startTime: str = ""


class DisplayData(BaseModel):
    teamId: int
    leagueId: int = 0
    abbreviation: str = ""
    trackedTeam: str = ""
    wins: int = 0
    losses: int = 0
    gamesBehind: str = "----"  # string: MLB returns "-" for the division leader
    last10: str = ""
    hasCurrentGame: bool = False
    # Both are always present (one default-filled) so the device parser needs no
    # conditional structure; hasCurrentGame selects which to read.
    current: CurrentGame = CurrentGame()
    next: NextGame = NextGame()


# --- The four fetches, ported from net.cpp ----------------------------------

# fields= filters kept verbatim from net.cpp: known-working, and the standings
# one carries the "every key on a path must be listed" trap (net.cpp:129-131).
TEAM_FIELDS = "teams,league,id,name,abbreviation"
STANDINGS_FIELDS = (
    "records,teamRecords,team,id,gamesBack,leagueRecord,wins,losses,splitRecords,type"
)
SCHEDULE_FIELDS = "dates,games,gamePk,status,abstractGameState,detailedState"
GAME_FIELDS = (
    "gameData,status,abstractGameState,detailedState,"
    "teams,home,away,name,abbreviation,"
    "datetime,time,ampm,"
    "liveData,linescore,currentInning,isTopInning,outs,balls,strikes,"
    "runs,offense,defense,batter,pitcher,fullName,id,first,second,third,"
    "boxscore,players,person,stats,batting,hits,atBats,pitching,numberOfPitches"
)

TTL_TEAM = 24 * 60 * 60
TTL_STANDINGS = 10 * 60
TTL_SCHEDULE = 60
TTL_GAME = 5

# States that linger in the schedule but won't be played as scheduled.
SKIP_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended"}


async def fetch_team_info(team_id: int) -> dict:
    """-> {leagueId, trackedTeam, abbreviation}. 404 if the team is unknown."""
    url = f"{BASE}/api/v1/teams/{team_id}?fields={TEAM_FIELDS}"
    doc = await fetch_json(url, TTL_TEAM, not_found_is_404=True)
    teams = doc.get("teams") or []
    if not teams:  # defensive: MLB 404s for an unknown id (handled above)
        raise HTTPException(status_code=404, detail=f"unknown teamId {team_id}")
    t = teams[0]
    return {
        "leagueId": t.get("league", {}).get("id", 0),
        "trackedTeam": t.get("name", ""),
        "abbreviation": t.get("abbreviation", ""),
    }


async def fetch_standings(league_id: int, team_id: int) -> dict:
    """-> {wins, losses, gamesBehind, last10}, or {} if the team isn't found."""
    season = datetime.now(timezone.utc).year
    url = (
        f"{BASE}/api/v1/standings?leagueId={league_id}&season={season}"
        f"&standingsTypes=regularSeason&fields={STANDINGS_FIELDS}"
    )
    doc = await fetch_json(url, TTL_STANDINGS)

    for rec in doc.get("records", []):
        for tr in rec.get("teamRecords", []):
            if tr.get("team", {}).get("id") != team_id:
                continue
            out = {
                "gamesBehind": tr.get("gamesBack", "----"),  # keep as string ("-" = leader)
                "wins": tr.get("leagueRecord", {}).get("wins", 0),
                "losses": tr.get("leagueRecord", {}).get("losses", 0),
                "last10": "",
            }
            for sr in tr.get("records", {}).get("splitRecords", []):
                if sr.get("type") == "lastTen":
                    out["last10"] = f"{sr.get('wins', 0)}-{sr.get('losses', 0)}"
                    break
            return out
    return {}


def _schedule_date(offset_days: int) -> str:
    # UTC to match scheduleDate()'s gmtime_r in net.cpp:161-168.
    d = datetime.now(timezone.utc).date() + timedelta(days=offset_days)
    return d.strftime("%Y-%m-%d")


async def fetch_game_ids(team_id: int) -> dict:
    """-> {'prev','current','next'} gamePks (0 if none). ±10-day UTC window."""
    url = (
        f"{BASE}/api/v1/schedule?sportId=1&teamId={team_id}"
        f"&startDate={_schedule_date(-10)}&endDate={_schedule_date(10)}"
        f"&fields={SCHEDULE_FIELDS}"
    )
    doc = await fetch_json(url, TTL_SCHEDULE)

    ids = {"prev": 0, "current": 0, "next": 0}
    # Dates ascending, games within a date ascending.
    for date in doc.get("dates", []):
        for g in date.get("games", []):
            status = g.get("status", {})
            if status.get("detailedState") in SKIP_STATES:
                continue
            state = status.get("abstractGameState")
            game_pk = g.get("gamePk", 0)
            if state == "Live":
                ids["current"] = game_pk  # last Live seen
            elif state == "Final":
                ids["prev"] = game_pk  # last Final seen
            elif state == "Preview" and ids["next"] == 0:
                ids["next"] = game_pk  # first Preview seen
    return ids


def _player_stats(box_teams: dict, person_id: int, group: str) -> dict:
    """Day stats for a player from the boxscore, keyed by dynamic 'ID<n>' keys."""
    for side in ("home", "away"):
        for player in box_teams.get(side, {}).get("players", {}).values():
            if player.get("person", {}).get("id") == person_id:
                return player.get("stats", {}).get(group, {}) or {}
    return {}


async def fetch_game_data(team_id: int) -> dict:
    """-> {hasCurrentGame, current?, next?}. Empty dict when there's no game.

    Live-feed failures propagate as HTTPException(502); the caller decides
    whether to degrade to standings-only.
    """
    ids = await fetch_game_ids(team_id)
    has_current = ids["current"] != 0
    game_id = ids["current"] if has_current else ids["next"]
    if game_id == 0:
        # No live or upcoming game (off-season, or a gap wider than the window).
        # Skip the live feed entirely; the device still renders the standings panel.
        return {"hasCurrentGame": False}

    url = f"{BASE}/api/v1.1/game/{game_id}/feed/live?fields={GAME_FIELDS}"
    doc = await fetch_json(url, TTL_GAME)

    gd = doc.get("gameData", {})
    ld = doc.get("liveData", {})
    teams = gd.get("teams", {})

    def team_identity(side: str) -> dict:
        t = teams.get(side, {})
        return {"name": t.get("name", ""), "abbrev": t.get("abbreviation", "")}

    home = team_identity("home")
    away = team_identity("away")

    if has_current:
        ls = ld.get("linescore", {})
        off = ls.get("offense", {})
        defense = ls.get("defense", {})
        ls_teams = ls.get("teams", {})

        batter = off.get("batter", {})
        pitcher = defense.get("pitcher", {})
        box_teams = ld.get("boxscore", {}).get("teams", {})
        bat_stats = _player_stats(box_teams, batter.get("id", 0), "batting")
        pit_stats = _player_stats(box_teams, pitcher.get("id", 0), "pitching")

        current = {
            "gameId": game_id,
            "home": {**home, "score": ls_teams.get("home", {}).get("runs", 0)},
            "away": {**away, "score": ls_teams.get("away", {}).get("runs", 0)},
            "batter": {
                "name": batter.get("fullName", ""),
                "hits": bat_stats.get("hits", 0),
                "atBats": bat_stats.get("atBats", 0),
            },
            "pitcher": {
                "name": pitcher.get("fullName", ""),
                "pitchCount": pit_stats.get("numberOfPitches", 0),
            },
            "isTopInning": ls.get("isTopInning", True),
            "inning": ls.get("currentInning", 1),
            "outs": ls.get("outs", 0),
            "strikes": ls.get("strikes", 0),
            "balls": ls.get("balls", 0),
            # Base occupancy is presence-of-key: the base key is absent when empty.
            "runners": {
                "first": "first" in off,
                "second": "second" in off,
                "third": "third" in off,
            },
        }
        return {"hasCurrentGame": True, "current": current}

    dt = gd.get("datetime", {})
    time_str = dt.get("time", "")
    ampm = dt.get("ampm", "")
    start_time = f"{time_str} {ampm}" if ampm else time_str
    return {
        "hasCurrentGame": False,
        "next": {"home": home, "away": away, "startTime": start_time},
    }


# --- Endpoint ---------------------------------------------------------------


@app.get("/teamInfo/{team_id}", response_model=DisplayData)
async def team_info(team_id: int) -> DisplayData:
    # Team info and schedule are independent; fetch concurrently. Standings needs
    # leagueId from team info; the live feed needs the gamePk from the schedule.
    team_task = asyncio.create_task(fetch_team_info(team_id))
    game_task = asyncio.create_task(fetch_game_data(team_id))

    # Team info gates everything renderable, so let its errors (incl. 404) surface.
    team = await team_task

    standings = await fetch_standings(team["leagueId"], team_id)

    try:
        game = await game_task
    except HTTPException as exc:
        # Live-feed failure degrades to standings-only (200) rather than failing
        # the whole request; net.cpp ignores fetch return values and leaves the
        # panel stale rather than blank.
        log.warning("game data unavailable for teamId %s: %s", team_id, exc.detail)
        game = {"hasCurrentGame": False}

    data = DisplayData(
        teamId=team_id,
        leagueId=team["leagueId"],
        abbreviation=team["abbreviation"],
        trackedTeam=team["trackedTeam"],
        **standings,
    )
    data.hasCurrentGame = game.get("hasCurrentGame", False)
    if "current" in game:
        data.current = CurrentGame(**game["current"])
    if "next" in game:
        data.next = NextGame(**game["next"])
    return data
