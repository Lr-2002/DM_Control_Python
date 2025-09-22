import time
from src.usb_class import usb_class

from usb_hw_wrapper import USBHardwareWrapper
usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")
usb_hw = USBHardwareWrapper(usb_hw)
