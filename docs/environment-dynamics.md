# ChargingEnv Dynamics

This document describes the transition dynamics of `ChargingEnv` defined in
`Chargym_Charging_Station/envs/Charging_Station_Enviroment.py`.

The environment simulates a single day (24 hourly timesteps, `t = 0..23`) at an
EV charging station with 10 parking slots, a PV array, and a grid connection.

---

## 1. Observation (State) Space

At each timestep `t` the agent observes a vector of dimension `8 + 2*N_cars = 28`:

| Slice | Meaning | Normalisation |
|---|---|---|
| `[0]` | Current solar radiation | `Radiation[t] / 1000` |
| `[1]` | Current electricity price | `Price[t] / 0.1` |
| `[2:8]` | 3-step-ahead predictions of radiation and price (interleaved) | same as above |
| `[8:18]` | Battery SOC of each car (`Battery[car]`) | raw value in [0, 1] |
| `[18:28]` | Normalised time-to-departure of each car (`Departure_hour[car] / 24`) | [0, 1] |

---

## 2. Solar Radiation and Electricity Price Evolution

### 2.1 Solar radiation

Source: `Weather.mat` contains minute-resolution weather data (`mydata`).
During `reset()`, `Energy_Calculations.Energy_Calculation` averages each 60-minute
window to produce hourly values for `24 * (number_of_days + 1) = 48` hours (one
extra day of buffer for the 3-step lookahead at `t = 23`).

The raw hourly radiation is stored in `Radiation[day, t]` (W/m^2).
PV generation is derived deterministically:

```
Renewable[day, t] = Radiation[day, t] * 1.5 * (PV_Surface * PV_effic / 1000) * solar_flag
```

where `PV_Surface = 2.279 * 1.134 * 20 m^2`, `PV_effic = 0.21`, and `solar_flag`
is 0 or 1.

**Key point:** radiation and renewable generation are loaded once at `reset()` and
remain fixed for the entire episode. They are **not** resampled at each timestep --
the 24-hour profile is deterministic given the Weather.mat file.

### 2.2 Electricity price

Prices are **hard-coded** 24-element profiles selected by `price_flag` (1--4).
The chosen 24-element profile is concatenated with itself to form a 48-element
array (providing the lookahead buffer), and this same profile is reused every
day.

| `price_flag` | Pattern |
|---|---|
| 1 | 0.05 off-peak (hours 0--6, 20--23), 0.10 peak (hours 7--19) |
| 2 | Smooth ramp up 6am--10am, plateau 0.10, shoulder 5--7pm, taper |
| 3 | High-variability profile, peaks at hours 13 and 17 |
| 4 | Two-tier: early-morning 0.10, midday 0.10, evening 0.10, off-peak 0.05 |

**Key point:** prices are fully deterministic and identical across episodes.

### 2.3 Independence of radiation and price

Solar radiation comes from a weather trace file; electricity prices come from a
hard-coded array. They are **generated independently** -- there is no coupling or
correlation mechanism between them. The 3-step predictions exposed in the
observation are the true future values (perfect forecast), not noisy samples.

---

## 3. EV Fleet Initialisation (at `reset()`)

`Init_Values.InitialValues_per_day` generates stochastic arrival/departure
schedules and initial SOC for each car using the environment's seeded RNG
(`self.np_random`).

### 3.1 Arrival process

For each car, at each hour `t = 0..23` (scanned sequentially):

```python
if car_not_present:
    arrival = round(np_random.rand() - 0.1)   # P(arrival) ~ 0.4
```

> `np_random.rand()` is Uniform[0, 1].  `round(x - 0.1)` returns 1 when
> `x >= 0.6` (since `round(0.5) = 0` in Python's banker's rounding and
> `round(y) = 1` for `y >= 0.5`, so we need `x - 0.1 >= 0.5`, i.e. `x >= 0.6`).
> Therefore **P(arrival = 1) = 0.4** per hour per absent car.

An arrival only materialises if `hour <= 20` (latest arrival at 8 pm).
On arrival the initial SOC is sampled:

```
BOC[car, hour] = np_random.randint(20, 50) / 100    # uniform in {0.20, 0.21, ..., 0.49}
```

### 3.2 Departure scheduling

Immediately upon arrival at hour `h`, a departure hour is sampled:

```
upper_limit = min(h + 10, 25)
departure = np_random.randint(h + 4, upper_limit)   # stay duration in [4, min(10, 25-h)-1] hours
```

The car is then present for hours `[h, h+1, ..., departure-1]` and departs at
hour `departure`.

### 3.3 Multiple visits

A car can visit the station **more than once** per day. After departing, it
re-enters the hourly arrival lottery. Each visit gets its own arrival hour,
departure hour, and initial SOC.

---

## 4. State-of-Charge (SOC / BOC) Dynamics

`BOC` is a `[N_cars x 25]` matrix (25 columns to accommodate the `t+1` write at
`t = 23`).

### 4.1 When a car is present

The agent's action `a[car] in [-1, 1]` is converted to a charging power:

```python
if a[car] >= 0:                            # charging
    max_energy = min(10, (1 - BOC[car,t]) * EV_capacity)
else:                                       # discharging (V2G)
    max_energy = min(10, BOC[car,t] * EV_capacity)

P_charging[car] = a[car] * max_energy      # kW, signed
```

where `EV_capacity = 30 kWh` and the 10 kW cap reflects the charger rating.
The SOC update is:

```
BOC[car, t+1] = BOC[car, t] + P_charging[car] / EV_capacity
```

This means `P_charging` is in kWh-per-timestep (since each timestep is 1 hour,
kW and kWh coincide numerically).

### 4.2 When a car is absent (`present_cars[car, t] == 0`)

`P_charging[car]` is forced to 0 regardless of the action, and
**`BOC[car, t+1]` is not written**. Because `BOC` is initialised with
`np.zeros`, any column that is never written stays at 0.

Concretely, if a car departs at hour `d` with SOC 0.8:

- At `t = d-1` (last present step): action writes `BOC[car, d] = 0.8`.
- At `t = d` (first absent step): observation reads `BOC[car, d] = 0.8`, but
  no write to `BOC[car, d+1]` occurs, so it remains 0 (from `np.zeros`).
- At `t = d+1` onward: observation reads `BOC[car, d+1] = 0`.

**The SOC persists for one timestep after departure, then drops to 0.**
Before the first arrival, SOC is likewise 0.

### 4.3 When a car arrives

At arrival hour `h`, `BOC[car, h]` is set to the sampled initial SOC
(0.20--0.49) during `reset()`. From `h` onward (while present), the SOC evolves
via the charging equation above.

### 4.4 When a car departs

A car is flagged as *leaving* at timestep `t` if it is present at `t` **and**
`t + 1` is in its departure list. The departure penalty is computed on
`BOC[car, t+1]` (i.e., after the last charging action has been applied):

```
penalty_car = (2 * (1 - BOC[car, t+1]))^2
```

After departure, `present_cars[car, t'] = 0` for `t' >= departure`, so the SOC
freezes and the action has no further effect.

### 4.5 Summary table

| Event | SOC behaviour |
|---|---|
| Before first arrival | `BOC = 0` (from `np.zeros` init) |
| On arrival at hour `h` | `BOC[car, h]` set to sampled value in [0.20, 0.49] |
| While present | `BOC[car, t+1] = BOC[car, t] + a[car] * max_energy / 30` |
| Departure hour `d` (first absent step) | `BOC[car, d]` still holds the last written value (e.g. 0.8) |
| `d+1` onward | `BOC[car, d+1..] = 0` (never written, stays at `np.zeros` init) |

---

## 5. Time-to-Departure Dynamics

Time-to-departure is computed each timestep in `Simulate_Station3.Simulate_Station`:

```python
for car in range(N_cars):
    if present_cars[car, t] == 0:
        Departure_hour[car] = 0
    else:
        # find earliest future departure for this car
        for d in Departure[car]:
            if t < d:
                Departure_hour[car] = d - t
                break
```

### 5.1 When a car is present

`Departure_hour[car]` counts down by 1 each timestep:

```
t:   Departure_hour = d - t
t+1: Departure_hour = d - (t+1) = (d - t) - 1
```

It reaches 1 at the last timestep before departure (the car departs at `d`, so
at `t = d - 1` the remaining time is 1).

### 5.2 When a car is absent

`Departure_hour[car] = 0`. This applies both before the car's first arrival and
after all departures. There is no distinction between "not yet arrived" and
"already left" in this signal.

### 5.3 On arrival

At the arrival hour `h`, the car becomes present and `Departure_hour[car]` jumps
from 0 to `departure - h` (the full scheduled stay duration, 4--10 hours).

### 5.4 On departure

At `t = departure - 1` (the last hour the car is present), `Departure_hour = 1`.
At `t = departure`, `present_cars = 0`, so `Departure_hour` drops to 0.

### 5.5 In the observation

The agent sees `Departure_hour[car] / 24`, so values range from 0 (absent or
departing now) to roughly 0.42 (10-hour stay, just arrived).

### 5.6 Summary table

| Event | Time-to-departure |
|---|---|
| Before arrival | 0 |
| On arrival (hour `h`, departure `d`) | `d - h` (jumps up) |
| While present, each step | decrements by 1 |
| Last present step (`t = d-1`) | 1 |
| After departure | 0 |

---

## 6. Reward Structure

The reward returned to the agent is `-Cost`, where:

```
Cost = Cost_grid + Cost_departure

Cost_grid     = max(0, Total_charging - RES_available) * Price[t]
Cost_departure = sum over departing cars: (2 * (1 - BOC[car, t+1]))^2
```

- **Grid cost**: only the portion of charging demand exceeding available PV
  output incurs cost. Discharging (negative `Total_charging`) can offset demand
  from other cars but `Grid_final` is floored at 0 (no feed-in revenue).
- **Departure penalty**: quadratic in the SOC deficit. A car leaving at 100% SOC
  pays 0; a car leaving at 0% SOC pays 4.0.
- A wasted-renewables penalty (`Cost_2`) is defined in code but **commented out**.

---

## 7. Episode Termination

The episode ends when `timestep` reaches 24. The environment sets `done = True`
and saves results to `Results.mat`. A new episode begins with `reset()`, which
re-samples the entire EV fleet schedule (unless `reset_flag=1` to reload a saved
schedule).
