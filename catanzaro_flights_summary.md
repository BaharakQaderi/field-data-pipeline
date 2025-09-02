# Catanzaro Flights Enhanced Analysis Summary

## Flight Overview

| Flight | Date | Duration | POD Brake Status | Data Points |
|--------|------|----------|------------------|-------------|
| Flight_5 | 2025-07-29 | 6m 13s | 0.00% engaged | 966 OPC |
| Flight_6 | 2025-07-29 | 7m 5s | 0.00% engaged | 1090 OPC |
| Flight_7 | 2025-07-29 | 2m 29s | 0.00% engaged | 381 OPC |

## Enhanced Power Metrics

### PoMecD = RPMd/60 * π * 0.491 * Ftot * 9.81

| Flight | Mean | Max | Min |
|--------|------|-----|-----|
| Flight_5 | 90.20 | 6751.63 | -6602.31 |
| Flight_6 | 414.08 | 11383.43 | -6372.12 |
| Flight_7 | -961.22 | 2976.17 | -8696.45 |

### PoMecGen = Vi / 60 * 2 * π * Td

| Flight | Mean | Max | Min |
|--------|------|-----|-----|
| Flight_5 | 66.82 | 1364.61 | -341.50 |
| Flight_6 | 164.10 | 2423.25 | -932.58 |
| Flight_7 | -64.85 | 1374.27 | -1361.08 |

### PoBatt = OPC_CONV_MEAS_FB_LS_PWR_SCALED_CALC

| Flight | Mean | Max | Min |
|--------|------|-----|-----|
| Flight_5 | -1.28 | 0.22 | -9.38 |
| Flight_6 | -2.01 | 3.10 | -11.24 |
| Flight_7 | -0.93 | 5.19 | -9.50 |

## Field Mapping

The following OPC fields were successfully extracted and used:

- **RPMd** → `OPC_DsEncoder.outTamburo_SpeedRPM`
- **Td** → `OPC_DsInverters.Torque_ActualValue[2]`
- **Vi** → `OPC_DsInverters.Velocity_ActualValue[2]`
- **Pinv** → `OPC_DsInverters.Power[2]`
- **Pbatt** → `OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC`
- **Ftot** → `OPC_DsLoadCells.MeasureFloat_SUM`

## Files Generated

For each flight (5, 6, 7), the following enhanced files were created:

1. **enhanced_flight_timeseries.png** - Updated visualizations with:
   - Flight duration in plot title
   - POD brake status percentage
   - New power metrics plots (PoMecD, PoMecGen, PoBatt)
   - Enhanced generator torque vs RPM analysis

2. **enhanced_flight_metrics.txt** - Complete metrics including:
   - Mean, Max, Min values for all three power metrics
   - Individual field statistics (RPMd, Td, Vi, Pinv, Pbatt, Ftot)

3. **opc_data_enhanced.csv** - Original OPC data plus new calculated columns:
   - PoMecD column
   - PoMecGen column  
   - PoBatt column

## Key Observations

1. **Flight Duration**: Flight_6 was the longest (7m 5s), Flight_7 was shortest (2m 29s)
2. **Brake Usage**: All three flights had 0% brake engagement throughout
3. **Power Generation**: Flight_6 showed highest PoMecGen mean (164.10), Flight_7 showed negative mean (-64.85)
4. **Mechanical Power**: Flight_6 had highest PoMecD mean (414.08), Flight_7 was negative (-961.22)
5. **Battery Power**: All flights showed negative battery power (energy being consumed rather than generated)