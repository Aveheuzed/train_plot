This repo contains accelerometer data I took during a train travel using my phone's sensors,
and a filtering/massaging/vizualization script.

The application used to collect the data was [Physics Toolbox Sensor Suite](https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite).

The train travelled along this line: [Angers -> Cholet](https://www.openstreetmap.org/relation/1633105).

__sensor details and setup:__

* The sensor is an [LSM6DSO](https://www.st.com/en/mems-and-sensors/lsm6dso.html) according to the app.

* Orientation of the sensor relative to the train's motion:
  * x pointing left
  * y pointing back
  * z pointing up

All measurements are in `g` (standard Earth gravity pull), assumed here to be 9.81 m.s^-2.
Therefore, at rest, the sensor theoretically yields `(0; 0; 1)`.
