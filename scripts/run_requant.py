import sys
import traceback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
try:
    from quantize_zaya_ct_nvfp4 import main
    main()
except SystemExit:
    raise
except:
    traceback.print_exc()
    sys.exit(1)
