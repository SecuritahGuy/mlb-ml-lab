# Feature Catalog

All features produced by registered `FeatureExtractor`s in
`src/mlb_ml_lab/features/`. The feature matrix rows are merged by
`(player_id, game_pk, date)` — each extractor adds its columns to
the same row.

27 extractors produce ~132 feature columns.

---

## 1. BullpenQualityFeatures (`bullpen`)

Opponent reliever-only pitching stats, IP-weighted from individual
pitcher season stats.

| Column | Type | Description |
|--------|------|-------------|
| `bullpen_era` | float | Opponent bullpen ERA |
| `bullpen_k_per_9` | float | Opponent bullpen K/9 |
| `bullpen_whip` | float | Opponent bullpen WHIP |
| `bullpen_ba_against` | float | Opponent bullpen BAA |
| `bullpen_hr_per_9` | float | Opponent bullpen HR/9 |

**Source**: `MlbClient.get_team_bullpen_stats()`

---

## 2. GamePaceFeature (`game_pace`)

Opponent's average game pace (duration and pitches per game).

| Column | Type | Description |
|--------|------|-------------|
| `opp_pace_time_per_game` | float | Opponent avg game duration (min) |
| `opp_pace_pitches_per_game` | float | Opponent avg pitches per game |

**Source**: `MlbClient.get_game_pace()`

---

## 3. HomeAwayFeature (`context`)

| Column | Type | Description |
|--------|------|-------------|
| `is_home` | int (0/1) | 1 if game at player's home stadium |

**Source**: Game log

---

## 4. InjuryFeatures (`injuries`)

Player IL status derived from MLB Stats API transactions.

| Column | Type | Description |
|--------|------|-------------|
| `days_on_il` | float or None | Consecutive days on IL (None if active) |
| `days_since_il` | float or None | Days since last IL activation (None if on IL) |
| `il_flag` | int (0/1) | 1 if on IL at game date |

**Source**: `MlbClient.get_transactions(season)`, processed by
`build_player_timelines()`.

**Backtest**: Δ AUC -0.07 bps (no impact). PA≥50 filter already
removes chronically injured players.

---

## 5. LeagueContextFeatures (`league`)

Season-level league-average context.

| Column | Type | Description |
|--------|------|-------------|
| `league_avg` | float | League BA |
| `league_obp` | float | League OBP |
| `league_slg` | float | League SLG |
| `league_ops` | float | League OPS |
| `league_runs_per_game` | float | League runs per game |

**Source**: Computed from team hitting stats

---

## 6. MonthlyTeamPitchingFeatures (`matchup`)

Opponent pitching splits aggregated through the month *before* the
game date (no lookahead).

| Column | Type | Description |
|--------|------|-------------|
| `mth_opp_era` | float | Opponent ERA through prior month |
| `mth_opp_k_per_9` | float | Opponent K/9 through prior month |
| `mth_opp_whip` | float | Opponent WHIP through prior month |
| `mth_opp_ba_against` | float | Opponent BAA through prior month |
| `mth_opp_games` | int | Games in monthly aggregation |

**Source**: `MlbClient.get_team_pitching_monthly_stats()`

---

## 7. OddsFeatures (`odds`)

Moneyline odds and implied win probabilities.

| Column | Type | Description |
|--------|------|-------------|
| `team_ml` | int | Team moneyline |
| `opp_ml` | int | Opponent moneyline |
| `team_implied_prob` | float | Team implied win prob |
| `opp_implied_prob` | float | Opponent implied win prob |

**Source**: SBR odds scraper (`sportsbookreview.com`)

---

## 8. ParkFactorFeatures (`context`)

Ballpark adjustment ratios from Baseball Savant.

| Column | Type | Description |
|--------|------|-------------|
| `park_wOBA` | float | Park factor for wOBA (1.0 = neutral) |
| `park_HR` | float | Park factor for HR |
| `park_1B` | float | Park factor for singles |

**Source**: `ParkFactors` (scrapes Baseball Savant park factors page)

---

## 9. PlayerQualityFeatures (`player`)

Player demographics and weighted career stats (last 3 seasons).

| Column | Type | Description |
|--------|------|-------------|
| `player_age` | float | Age in years at game date |
| `years_experience` | float | Years since MLB debut |
| `bats_right` | int (0/1) | 1 if right-handed batter |
| `bats_left` | int (0/1) | 1 if left-handed batter |
| `throws_right` | int (0/1) | 1 if right-handed thrower |
| `throws_left` | int (0/1) | 1 if left-handed thrower |
| `position_cat` | int (0-3) | 0=IF, 1=OF, 2=C, 3=DH |
| `career_avg` | float | Weighted career BA |
| `career_obp` | float | Weighted career OBP |
| `career_slg` | float | Weighted career SLG |
| `career_ops` | float | Weighted career OPS |
| `career_hr` | int | Weighted career HR |

**Source**: `MlbClient.get_player()`, `MlbClient.get_player_season_stats()`

---

## 10. RestDaysFeature (`context`)

| Column | Type | Description |
|--------|------|-------------|
| `rest_days` | int | Days since player's last game |

**Source**: Computed from game logs

---

## 11. RollingAdvancedMetrics (`game_log`)

Rolling advanced hitting metrics over fixed game windows.

Window sizes: 10, 20. All columns follow pattern `{metric}_{window}`.

| Column | Description |
|--------|-------------|
| `rolling_avg_{w}` | Batting average |
| `rolling_obp_{w}` | On-base percentage |
| `rolling_slg_{w}` | Slugging percentage |
| `rolling_ops_{w}` | OPS |
| `rolling_iso_{w}` | Isolated power (SLG - AVG) |
| `rolling_babip_{w}` | BABIP |
| `rolling_bb_pct_{w}` | Walk rate |
| `rolling_k_pct_{w}` | Strikeout rate |
| `rolling_woba_{w}` | wOBA |
| `rolling_ops_plus_{w}` | OPS+ (park-adjusted, league-normalized) |
| `rolling_wrc_plus_{w}` | wRC+ (park-adjusted, league-normalized) |

22 columns total.

---

## 12. RollingBABIP (`game_log`)

| Column | Type | Description |
|--------|------|-------------|
| `babip_last_20` | float | BABIP over last 20 games |

---

## 13. RollingHits (`game_log`)

Raw hit counts and rates over game windows.

| Column | Description |
|--------|-------------|
| `hits_last_{5,10,20}` | Total hits in window |
| `hit_rate_last_{5,10,20}` | Hits per game in window |

6 columns.

---

## 14. RollingOpponentPitching (`matchup`)

Opponent rolling pitching rates from game logs *before* the game date
(no lookahead bias).

| Column | Type | Description |
|--------|------|-------------|
| `rolling_opp_k_rate` | float | Opponent K per PA |
| `rolling_opp_ba_against` | float | Opponent BAA |
| `rolling_opp_walk_rate` | float | Opponent BB per PA |
| `rolling_opp_sample_games` | int | Games in sample |

---

## 15. RollingPlateAppearances (`game_log`)

Plate appearance and K/BB rates over game windows.

| Column | Description |
|--------|-------------|
| `avg_pa_last_{10,20}` | Avg PA per game |
| `bb_rate_last_{10,20}` | BB rate |
| `k_rate_last_{10,20}` | K rate |

6 columns.

---

## 16. RollingStatcastFeatures (`statcast_search`)

Rolling statcast batted-ball metrics from pitch-by-pitch data.

Window sizes: 10, 20. All columns follow pattern `sc_{metric}_{w}`.

| Column | Description |
|--------|-------------|
| `sc_avg_ev_{w}` | Avg exit velocity |
| `sc_hardhit_rate_{w}` | Hard hit rate (≥95 mph) |
| `sc_barrel_rate_{w}` | Barrel rate |
| `sc_avg_la_{w}` | Avg launch angle |
| `sc_sweet_spot_rate_{w}` | Sweet-spot contact rate (8-32°) |
| `sc_avg_xba_{w}` | Avg xBA per BBE |
| `sc_avg_xwoba_{w}` | Avg xwOBA per BBE |
| `sc_avg_distance_{w}` | Avg batted ball distance (ft) |
| `sc_fbld_rate_{w}` | Fly ball + line drive rate |
| `sc_gb_rate_{w}` | Ground ball rate |
| `sc_bbe_count_{w}` | Batted ball events in window |

22 columns total.

**Source**: `MlbClient.get_statcast_search_data()`

---

## 17. ScheduleDensityFeatures (`schedule`)

Opponent rest and schedule density.

| Column | Type | Description |
|--------|------|-------------|
| `opp_rest_days` | int | Days since opponent's last game |
| `opp_games_last_5` | int | Games opponent played in last 5 days |
| `opp_games_last_10` | int | Games opponent played in last 10 days |
| `opp_games_last_14` | int | Games opponent played in last 14 days |

**Source**: `MlbClient.get_season_schedule()`

---

## 18. StartingPitcherFeatures (`pitching`)

Opposing starting pitcher quality and platoon matchup.

| Column | Type | Description |
|--------|------|-------------|
| `opp_pitcher_id` | int | MLBAM ID of opposing starter |
| `opp_pitcher_era` | float | Starter ERA |
| `opp_pitcher_k_per_9` | float | Starter K/9 |
| `opp_pitcher_whip` | float | Starter WHIP |
| `opp_pitcher_ba_against` | float | Starter BAA |
| `opp_pitcher_hr_per_9` | float | Starter HR/9 |
| `opp_pitcher_k_rate` | float | Starter K per PA |
| `opp_pitcher_bf_per_9` | float | Starter batters faced per 9 IP |
| `same_hand_advantage` | int (0/1) | 1 if same-handed (pitcher advantage) |

**Source**: `MlbClient.get_player_season_stats()` (group=pitching),
`MlbClient.get_player()` (for handedness)

---

## 19. StatcastAdvancedFeatures (`statcast`)

Season-level statcast hitting metrics from Baseball Savant leaderboard.

| Column | Type | Description |
|--------|------|-------------|
| `ba` | float | Actual BA |
| `slg` | float | Actual SLG |
| `woba` | float | Actual wOBA |
| `xba` | float | Expected BA (xBA) |
| `xwoba` | float | Expected wOBA |
| `xslg` | float | Expected SLG |
| `xba_diff` | float | xBA - BA |
| `xslg_diff` | float | xSLG - SLG |
| `xwoba_diff` | float | xwOBA - wOBA |
| `hardhit_percent` | float | Hard hit rate (≥95 mph) |
| `barrels_per_bbe_percent` | float | Barrels per BBE |
| `brl_pa` | float | Barrels per PA |
| `avg_hit_speed` | float | Avg exit velocity |
| `max_hit_speed` | float | Max exit velocity |
| `ev50` | float | Median exit velocity |
| `avg_launch_angle` | float | Avg launch angle |
| `anglesweetspotpercent` | float | Sweet-spot contact rate |
| `fbld` | float | Fly ball + line drive % |
| `gb` | float | Ground ball % |
| `avg_distance` | float | Avg batted ball distance (ft) |
| `max_distance` | float | Max batted ball distance (ft) |
| `avg_hr_distance` | float | Avg HR distance (ft) |
| `ev95plus` | int | Count of 95+ mph EV |
| `barrels` | int | Count of barrels |

24 columns.

**Source**: Baseball Savant CSV leaderboard (Statcast, Expected Stats)

---

## 20. StreaksFeature (`streaks`)

Current hitting and on-base streaks.

| Column | Type | Description |
|--------|------|-------------|
| `hitting_streak` | int | Consecutive games with ≥1 hit |
| `onbase_streak` | int | Consecutive games reaching base |

**Source**: `MlbClient.get_stats_streaks()` or computed from game logs

---

## 21. TeamDefenseFeatures (`matchup`)

Opponent team fielding performance.

| Column | Type | Description |
|--------|------|-------------|
| `opp_fielding_pct` | float | Team fielding percentage |
| `opp_errors` | int | Team total errors |
| `opp_double_plays` | int | Team double plays turned |

**Source**: `MlbClient.get_team_fielding_stats()`

**Backtest**: AUC 0.508 alone, Δ +0.05 bps additive. Negligible impact.

---

## 22. TeamLeadersFeature (`team_leaders`)

Opponent team's top hitter by BA, HR, RBI.

| Column | Type | Description |
|--------|------|-------------|
| `opp_top_avg` | float | Best BA on opponent team |
| `opp_top_hr` | int | Best HR total on opponent team |
| `opp_top_rbi` | int | Best RBI total on opponent team |

**Source**: `MlbClient.get_team_leaders()`

---

## 23. TeamPitchingFeatures (`matchup`)

Opponent team season-level pitching stats.

| Column | Type | Description |
|--------|------|-------------|
| `opp_era` | float | Team ERA |
| `opp_k_per_9` | float | Team K/9 |
| `opp_whip` | float | Team WHIP |
| `opp_ba_against` | float | Team BAA |
| `opp_hr_per_9` | float | Team HR/9 |

**Source**: `MlbClient.get_team_pitching_stats()`

---

## 24. TeamTrendFeatures (`game_log`)

Team-level rolling hit trends (from actual game results).

| Column | Type | Description |
|--------|------|-------------|
| `team_hits_last_5` | float | Team avg hits/game last 5 team games |
| `team_hits_last_10` | float | Team avg hits/game last 10 team games |
| `team_opp_hits_last_5` | float | Opponent avg hits/game last 5 games |
| `team_opp_hits_last_10` | float | Opponent avg hits/game last 10 games |

---

## 25. WeatherFeatures (`context`)

Historical weather at game time from MLB Stats API.

| Column | Type | Description |
|--------|------|-------------|
| `weather_condition` | str | Weather label (Clear, Cloudy, etc.) |
| `weather_temp` | float | Temperature (°F) |
| `weather_wind` | str | Wind description |

**Source**: MLB Stats API game feed (`get_game_context()`)

---

## 27. UmpireFeatures (`umpire`)

Home-plate umpire identity and experience proxy. HP umpire is the most
consequential official — their strike zone tendencies affect pitcher/batter
outcomes.

| Column | Type | Description |
|--------|------|-------------|
| `hp_umpire_id` | int | MLBAM ID of the home-plate umpire |
| `umpire_game_count` | int | Games this HP umpire has worked this season |

**Source**: MLB Stats API schedule hydration (`officials` hydrate on
`/schedule`), no extra API calls. Game counts come from
`MlbClient.count_umpire_games()`.

---

## 28. WeatherForecastFeatures (`forecast`)

Forecast weather from National Weather Service API.

| Column | Type | Description |
|--------|------|-------------|
| `forecast_temp` | float | Forecast temperature (°F) |
| `forecast_wind_speed` | str | Forecast wind speed |
| `forecast_wind_direction` | str | Forecast wind direction |
| `forecast_precip_pct` | float | Precipitation probability (0-100) |
| `forecast_conditions` | str | Forecast conditions label |

**Source**: NWS API (`NwsWeather`), called live during extraction.
