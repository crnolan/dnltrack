# Development notes

## Frame synchronisation

There are a few possibilities depending on how many GPIO lines are available via the M8 connector (see [this](https://discuss.luxonis.com/d/5243-oak-d-pro-w-poe-gpio-reference) forum post).

According to Jaka [here](https://discuss.luxonis.com/d/5243-oak-d-pro-w-poe-gpio-reference/7), only pin 1 of the M8 connector is GPIO.

How do we sync multiple cameras?

## TDT digital IO lines

It seems as though TDTs digital IO lines are set to output high at idle, which is a problem for the cameras as pin 1 held high at boot time kicks off the USB bootloader.