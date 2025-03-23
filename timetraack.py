import time
from datetime import datetime, timedelta, timezone

# Get the local time zone offset in seconds
local_offset_seconds = time.localtime().tm_gmtoff
local_offset = timedelta(seconds=local_offset_seconds)

# Get the current local time with offset
local_time = datetime.now(timezone(local_offset))

# Display the time and offset without fractional seconds
print(local_time.isoformat(timespec="seconds"))
