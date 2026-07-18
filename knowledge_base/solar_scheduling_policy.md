# Solar Scheduling Policy
UMORDA Energy Domain — Balcony Solar Panel Power Scheduling

## Direct Use Priority
Solar power should be used directly to power the home whenever solar output meets or exceeds current home consumption. Direct use avoids conversion losses that occur when energy is stored in a battery and then discharged. Using solar directly is always the first preference when sun is available and consumption matches output.

## Battery Storage Criteria
Surplus solar power beyond home consumption needs should be stored in the battery for later use, particularly for evening and night hours when panels generate nothing. Storage is the correct choice when solar output exceeds consumption and the battery has remaining capacity. Storing energy avoids the need to buy from the grid later at higher cost.

## Battery Full Rule
If the battery is already full (level 9), storing additional solar is not possible. In this case, surplus should either be used directly or sold to the grid if that option is available. Never attempt to store energy into a full battery.

## No Solar Situation
When solar output is zero (rain, heavy clouds, nighttime), the agent must not choose Use Solar Directly — there is no solar power to use. In this situation, the correct action is to discharge the battery if it has charge, or buy from the grid if the battery is empty.

## Time of Day Guidance
- Morning: moderate solar expected, moderate storage opportunity.
- Afternoon: peak solar generation, prioritize direct use and storage.
- Evening: solar declining, rely on stored battery energy.
- Night: no solar, battery is the only own-energy source.

## Grid Purchase Guideline
Buying from the grid should be a last resort when both solar output is zero and the battery is empty. Purchasing grid electricity when solar is available represents an inefficiency and increases electricity cost unnecessarily.
