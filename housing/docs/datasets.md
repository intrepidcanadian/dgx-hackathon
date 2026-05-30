# Toronto Open Data - Dataset Catalogue for DGX Spark Projects

Datasets discovered from [Toronto Open Data](https://open.toronto.ca/catalogue/), organized by refresh rate. Each dataset is tagged with the project(s) it supports:

- **S** = Shelter Intelligence (demand forecasting + resource routing)
- **F** = Food Safety Predictor (DineSafe risk classification)
- **M** = Mobility Intelligence (demand forecasting + anomaly detection)

---

## Real-Time Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Bike Share Station Status (GBFS) | Live bike/dock availability per station, 30s updates | | | M |
| Bike Share Station Information (GBFS) | Station locations, capacity, names | | | M |
| Road Restrictions | Active road closures, construction, events with lat/lng and severity | | | M |
| Fire Dispatch | Active fire/EMS dispatch events | | | |
| TTC Real-Time (NextBus) | Bus/streetcar GPS positions and predictions | | | M |

## Daily Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Shelter & Support Services Occupancy | **Primary target** - daily bed/room occupancy by shelter, program, sector | S | | |
| DineSafe | **Primary target** - restaurant inspections: pass/conditional/fail, deficiency details, fines | | F | |
| Non-Regulated Lead Water Samples | Lead levels (ppm) by partial postal code; 14K rows, thin but usable as enrichment | S | | |
| Outbreaks in Toronto Healthcare Institutions | Outbreak type, causative agent, institution, active status | S | | |
| Ferry Ticket Counts | 15-min interval sales/redemptions for Toronto Islands ferry; 272K+ rows | | | M |
| Cooling Centres / Warming Centres | Locations, hours, amenities, eligibility for emergency weather shelters | S | | |
| BodySafe | Body modification establishment inspections | | F | |
| COVID-19 Cases | Case-level data with demographics and neighbourhood | S | | |
| Daily Shelter & Overnight Service Usage | Alternate shelter feed with different groupings | S | | |

## Weekly Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Apartment Building Evaluation (RentSafeTO) | Building condition scores, violations, property standards | S | | |
| TTC Delay Data (Bus/Streetcar/Subway) | Delay incidents by route, station, cause, duration | | | M |
| Wellbeing Youth Survey | Youth indicators by neighbourhood | S | | |
| Municipal Code Complaints | Property standards and bylaw complaints | S | F | |

## Monthly Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Building Permits - Active/Cleared | Construction activity by location, type, value | S | F | M |
| 311 Service Requests | Citizen complaints/requests by type and location | S | F | |
| Short-Term Rental Registrations | Airbnb/rental listings; displacement and neighbourhood churn signals | S | F | |
| COVID-19 Wastewater Signal | Viral load trends in wastewater by treatment plant | S | | |
| Business Licences | Active licensed businesses by type and location | | F | |
| Municipal Licensing - Business Infractions | Licensing violations by business type | | F | |
| Automated Speed Enforcement Charges | Speed camera tickets by location | | | M |
| Red Light Camera Locations | Intersection enforcement cameras | | | M |
| Traffic Volumes (AADT) | Annual average daily traffic counts at intersections | | | M |
| Motor Vehicle Collisions (Monthly) | Collision data with location, severity, conditions | | | M |
| Parks Amenities & Facilities | Park infrastructure and locations | S | | M |
| Neighbourhood Profiles (Census) | Demographics, income, housing by neighbourhood | S | F | |
| Property Boundaries / Address Points | Geocoding and spatial reference | S | F | M |

## Semi-Annual Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Wellbeing Toronto Surveys (multiple) | Neighbourhood-level indicators: demographics, economics, health, civics | S | | |
| Property Tax Policy Rates | Tax rates by property class; economic pressure signal | | F | |
| Red Light Camera Annual Report | Camera effectiveness and violation trends | | | M |
| Ward Profiles | Demographics and service data by political ward | S | F | |
| Toronto Employment Survey | Jobs by sector and location | S | F | |
| School Locations (TDSB) | Public school locations and capacity | S | | |
| Child Care Centres | Licensed child care locations and capacity | S | | |
| Social Infrastructure Locations | Community services, libraries, rec centres | S | | |

## Annual Datasets

| Dataset | What It Adds | S | F | M |
|---------|-------------|---|---|---|
| Neighbourhood Crime Rates | Crime rates by neighbourhood and type (multi-year) | S | F | |
| Fire Incidents | Fire events by location, cause, casualties | | F | |
| Fire Services Emergency Incidents | Detailed incident data (124 resources) | S | F | |
| Subsidized Housing Listings | Subsidized housing supply and waitlists | S | | |
| Toronto Community Housing Data | Public housing stock, vacancies, conditions | S | | |
| Drop-In Locations (TDIN) | Drop-in centre locations (complementary to shelters) | S | | |
| HousingTO Action Plan | Housing policy targets and progress | S | | |
| Cost of Living in Toronto | Cost benchmarks for low-income households | S | | |
| Community Grants Allocations | Funding flows to social service organizations | S | | |
| Community Referrals by EMS (CREMS) | Paramedic referrals to community services; direct signal of street-level need | S | | |
| Paramedic Services Incident Data | EMS call volumes and response data | S | | |
| Land Ambulance Response Time | Response time standards and performance | S | | |
| Watermain Breaks | Infrastructure failure locations (water system age proxy) | S | | |
| Water Billing by Ward | Water consumption patterns by ward | | | |
| Small Business Property Tax Subclass | Eligible small business properties; turnover signal | | F | |
| Parking Tickets | Massive dataset (since 2008); spatial parking demand patterns | | | M |
| Police Traffic Collisions | Collision locations, causes, contributing factors | | | M |
| Bicycle Thefts | Bike theft locations and patterns | | | M |
| TTC Ridership Analysis | System-wide ridership trends | | | M |
| TTC Subway Station Usage | Station-level entry/exit counts | | | M |
| Traffic Signal Vehicle & Pedestrian Volumes | Intersection-level traffic and pedestrian counts | | | M |
| Travel Times - Bluetooth | Road segment travel times | | | M |
| Annual Energy Consumption (City Buildings) | Energy use by city building; activity proxy | | | M |
| Communicable Diseases Summary | Annual disease trends by type | S | | |
| Immunization Coverage for Students | Student vaccination rates by school/area | S | | |
| Budget - Operating by Expenditure Category | City operating budget allocations | S | F | |
| Licensed Dogs and Cats Reports | Pet licensing data | | | |

---

## Project Summaries

### Shelter Intelligence

**Goal:** Predict shelter occupancy 1-7 days ahead; identify which shelters will reach capacity; route people to available beds.

**Primary data:** Daily Shelter Occupancy (target variable: occupancy rate by shelter/program)

**Enrichment layers by frequency:**
| Layer | Refresh | Signal |
|-------|---------|--------|
| Shelter Occupancy | Daily | Target variable - beds/rooms occupied vs capacity |
| Weather (external API) | Daily | Cold snaps drive demand spikes |
| Outbreaks | Daily | Disease pressure on congregate settings |
| Cooling/Warming Centres | Daily | Overflow capacity during extreme weather |
| 311 Requests | Monthly | Neighbourhood distress indicators |
| Building Permits | Monthly | Construction displacement pressure |
| Short-Term Rentals | Monthly | Housing supply competition |
| COVID Wastewater | Monthly | Public health pressure |
| RentSafeTO | Weekly | Building condition / eviction risk |
| Wellbeing Surveys | Semi-annual | Demographic and economic context |
| Drop-In Locations | Annual | Complementary service network |
| Subsidized Housing | Annual | Housing supply baseline |
| Community Housing | Annual | Public housing stock context |
| Cost of Living | Annual | Economic pressure baseline |
| EMS Community Referrals | Annual | Street-level need signal |
| Crime Rates | Annual | Neighbourhood safety context |

**Model type:** Time-series forecasting (LSTM/Transformer) + classification (full/not-full)

---

### Toronto Food Safety Predictor

**Goal:** Predict DineSafe inspection outcomes (pass/conditional/fail); classify establishment risk by neighbourhood, cuisine type, and violation history.

**Primary data:** DineSafe inspections (target variable: inspection outcome)

**Enrichment layers by frequency:**
| Layer | Refresh | Signal |
|-------|---------|--------|
| DineSafe Inspections | Daily | Target variable - outcomes, deficiencies, fines |
| Business Licences | Monthly | Business type, age, licence status |
| Business Infractions | Monthly | Licensing violation history |
| 311 Requests | Monthly | Complaint patterns near establishments |
| Building Permits | Monthly | Construction near food establishments |
| Short-Term Rentals | Monthly | Neighbourhood commercial churn |
| Neighbourhood Profiles | Monthly | Demographics, income by neighbourhood |
| Property Tax Rates | Semi-annual | Economic pressure on businesses |
| Employment Survey | Semi-annual | Labour market conditions |
| Crime Rates | Annual | Neighbourhood quality signal |
| Fire Incidents | Annual | Commercial fire history by location |
| Small Business Tax | Annual | Business turnover signal |
| Operating Budget | Annual | Inspection funding levels |

**Model type:** Classification (risk score per establishment) + geospatial clustering

---

### Toronto Mobility Intelligence

**Goal:** Forecast demand across transport modes (bikes, ferry, TTC); detect anomalies from construction, events, weather.

**Primary data:** Bike Share real-time status + Ferry tickets + TTC delays

**Enrichment layers by frequency:**
| Layer | Refresh | Signal |
|-------|---------|--------|
| Bike Share Status | Real-time | Station-level bike/dock availability |
| Road Restrictions | Real-time | Active closures and construction |
| Ferry Tickets | Daily | 15-min demand patterns |
| TTC Delays | Weekly | Delay incidents by route and cause |
| Building Permits | Monthly | Construction disruption locations |
| Speed Enforcement | Monthly | Traffic enforcement patterns |
| Red Light Cameras | Monthly | Intersection enforcement |
| Motor Vehicle Collisions | Monthly | Collision hotspots |
| Traffic Volumes (AADT) | Semi-annual | Baseline traffic counts |
| TTC Ridership | Annual | System-wide demand trends |
| TTC Subway Usage | Annual | Station-level baselines |
| Traffic Signal Volumes | Annual | Intersection pedestrian/vehicle counts |
| Travel Times | Annual | Road segment speed baselines |
| Parking Tickets | Annual | Parking demand spatial patterns |
| Police Traffic Collisions | Annual | Historical collision patterns |
| Bicycle Thefts | Annual | Bike infrastructure risk |
| Energy Consumption | Annual | Building activity proxy |

**Model type:** Demand forecasting (time-series) + anomaly detection

---

## Data Access

**CKAN API base:** `https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/`

- List dataset metadata: `package_show?id={dataset-id}`
- Download CSV: `datastore/dump/{resource_id}?limit=N&format=csv`
- Search datasets: `package_search?q=&fq=refresh_rate:{rate}&rows=100`

**Specialty APIs:**
- Bike Share (GBFS): `https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status`
- Road Restrictions: `https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=json`
