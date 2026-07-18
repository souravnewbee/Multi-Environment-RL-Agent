# Battery Management Policy
UMORDA Energy Domain — Battery Storage Charge and Discharge Optimization

## Charge Decision Criteria
The battery should be charged when solar generation is strong and the battery has remaining capacity. Charging during peak solar hours (afternoon) is most efficient because it uses free solar energy rather than paid grid electricity. Charging from the grid is acceptable only when the grid price is cheap (price level 0) and the battery is low.

## Discharge Decision Criteria
The battery should be discharged to power the home when grid electricity is expensive (price level 2). Using stored battery energy during expensive grid periods avoids high electricity costs. Discharging is also appropriate when solar is unavailable and home consumption must be met from stored energy.

## Idle Decision Criteria
Keeping the battery idle is appropriate when solar output already covers home consumption without battery involvement. There is no benefit to charging or discharging when the home is energy-balanced from solar alone.

## Grid Price Thresholds
- Grid price 0 (cheap): good time to charge battery from grid if solar is unavailable and battery is low.
- Grid price 1 (normal): standard operation, charge from solar only.
- Grid price 2 (expensive): avoid buying from grid entirely, discharge battery to cover consumption.

## Battery Level Guidelines
- Battery level 0 to 1: critically low, avoid discharging further, charge immediately if possible.
- Battery level 2 to 4: low, prioritize charging when solar is available.
- Battery level 5 to 6: medium, normal operation based on price and solar.
- Battery level 7 to 8: high, discharge when grid is expensive.
- Battery level 9: full, do not attempt to charge further.

## Avoiding Battery Degradation
Repeatedly charging a full battery or discharging an empty battery causes degradation. The agent should avoid charging when battery is already full and avoid discharging when battery is at 0 or 1.
