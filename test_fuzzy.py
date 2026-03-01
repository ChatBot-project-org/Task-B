import sys
from io import StringIO
import mybot

sys.stdin = StringIO("Plastic bottle is 85% recyclable\nCheck certainty that plastic bottle is recyclable\n:quit\n")
mybot.main()
