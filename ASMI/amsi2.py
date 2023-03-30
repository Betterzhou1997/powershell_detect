#!/usr/bin/env python3

# > ------------------------------------

#       Antimalware Scan Interface

# > ------------------------------------

import sys
from enum import IntEnum
from ctypes import HRESULT, POINTER, windll, byref
from ctypes.wintypes import HANDLE, LPCWSTR, UINT, LPCSTR
from comtypes.hresult import S_OK

# > ------------------------------------------------------------------------------
#   The AMSI_RESULT enumeration specifies the types of results returned by scans
# > ------------------------------------------------------------------------------
"""
    typedef enum AMSI_RESULT {
        AMSI_RESULT_CLEAN,
        AMSI_RESULT_NOT_DETECTED,
        AMSI_RESULT_BLOCKED_BY_ADMIN_START,
        AMSI_RESULT_BLOCKED_BY_ADMIN_END,
        AMSI_RESULT_DETECTED
    } ;
"""


class AMSI_RESULT(IntEnum):
    """ AMSI Results Class """
    AMSI_RESULT_CLEAN = 0
    AMSI_RESULT_NOT_DETECTED = 1
    AMSI_RESULT_BLOCKED_BY_ADMIN_START = 16384
    AMSI_RESULT_BLOCKED_BY_ADMIN_END = 20479
    AMSI_RESULT_DETECTED = 32768


class Amsi(object):
    """ AMSI API Class """

    def __init__(self):
        # Inialize the context and session to utilize
        self.amsi_ctx = self.amsi_initialize()
        # self.amsi_ctx = S_OK
        self.amsi_session = self.amsi_open_session()

    # > -------------------------
    #   Initialize the AMSI API
    # > -------------------------
    """
        HRESULT AmsiInitialize(
            LPCWSTR      appName,
            HAMSICONTEXT *amsiContext
        );
    """

    def amsi_initialize(self):
        AmsiInitialize = windll.amsi.AmsiInitialize
        AmsiInitialize.argtypes = [LPCWSTR, POINTER(HANDLE)]  # Specify the argument data types
        AmsiInitialize.restype = HRESULT

        amsi_ctx = HANDLE(0)  # Return this context
        # amsi_ctx = POINTER(HANDLE)(HANDLE(0))  # Return this context
        amsi_hres = AmsiInitialize("amsi-test", byref(amsi_ctx))

        # If this function succeeds, it returns S_OK. Otherwise, it returns an HRESULT error code
        if amsi_hres != S_OK:
            print(f"[!]\tAmsiInitialize Error: {amsi_hres}")
            sys.exit()  # Exit if initialization fails

        return amsi_ctx

    # > ----------------------------------------------------------------------------------
    #   Remove the instance of the AMSI API that was originally opened by AmsiInitialize
    # > ----------------------------------------------------------------------------------
    """
        void AmsiUninitialize(
            HAMSICONTEXT amsiContext
        );
    """

    def amsi_uninitialize(self):
        AmsiUninitialize = windll.amsi.AmsiUninitialize
        AmsiUninitialize.argtypes = [HANDLE]
        AmsiUninitialize(self.amsi_ctx)

        return None

    # > -----------------------------------------------------------------------
    #   Opens a session within which multiple scan requests can be correlated
    # > -----------------------------------------------------------------------
    """
        HRESULT AmsiOpenSession(
            HAMSICONTEXT amsiContext,
            HAMSISESSION *amsiSession
        );
    """

    def amsi_open_session(self):
        AmsiOpenSession = windll.amsi.AmsiOpenSession
        AmsiOpenSession.argtypes = [HANDLE, POINTER(UINT)]  # Specify attribute data types
        AmsiOpenSession.restype = HRESULT

        amsi_session = UINT(0)  # Return this session
        amsi_hres = AmsiOpenSession(self.amsi_ctx, amsi_session)

        # If this function succeeds, it returns S_OK. Otherwise, it returns an HRESULT error code
        if amsi_hres != S_OK:
            print(f"[!]\tAmsiOpenSession Error: {amsi_hres}")
            sys.exit()  # Exit if session creation fails

        return amsi_session

    # > ----------------------------------------------------
    #   Close a session that was opened by AmsiOpenSession
    # > ----------------------------------------------------
    """
        void AmsiCloseSession(
            HAMSICONTEXT amsiContext,
            HAMSISESSION amsiSession
        );
    """

    def amsi_close_session(self):
        AmsiCloseSession = windll.amsi.AmsiCloseSession
        AmsiCloseSession.argtypes = [HANDLE, UINT]  # Specify attribute data types
        AmsiCloseSession.restype = HRESULT

        amsi_hres = AmsiCloseSession(self.amsi_ctx, self.amsi_session)

        return None

    # > ----------------------------
    #   Scans a string for malware
    # > ----------------------------
    """
        HRESULT AmsiScanString(
            HAMSICONTEXT amsiContext,
            LPCWSTR      string,
            LPCWSTR      contentName,
            HAMSISESSION amsiSession,
            AMSI_RESULT  *result
        );
    """

    def amsi_scan_string(self, data):
        AmsiScanString = windll.amsi.AmsiScanString
        AmsiScanString.argtypes = [HANDLE, LPCWSTR, LPCWSTR, UINT, POINTER(UINT)]  # Specify attribute data types
        AmsiScanString.restype = HRESULT

        amsi_res = UINT(0)  # Return this scan result
        amsi_hres = AmsiScanString(self.amsi_ctx, data, "string-data", self.amsi_session, byref(amsi_res))

        # If this function succeeds, it returns S_OK. Otherwise, it returns an HRESULT error code
        if amsi_hres != S_OK:
            print(f"[!]\tAmsiScanString Error: {amsi_hres}")
            sys.exit()  # Exit if scan fails

        return amsi_res

    # > --------------------------------------------
    #   Scans a buffer-full of content for malware
    # > --------------------------------------------
    """
        HRESULT AmsiScanBuffer(
            HAMSICONTEXT amsiContext,
            PVOID        buffer,
            ULONG        length,
            LPCWSTR      contentName,
            HAMSISESSION amsiSession,
            AMSI_RESULT  *result
        );
    """

    def amsi_scan_buffer(self, data):
        AmsiScanBuffer = windll.amsi.AmsiScanBuffer
        AmsiScanBuffer.argtypes = [HANDLE, LPCSTR, UINT, LPCWSTR, UINT, POINTER(UINT)]  # Specify attribute data types
        AmsiScanBuffer.restype = HRESULT

        amsi_res = UINT(0)  # Return this scan result
        amsi_hres = AmsiScanBuffer(self.amsi_ctx, data, len(data), "buffer-data", self.amsi_session, byref(amsi_res))

        # If this function succeeds, it returns S_OK. Otherwise, it returns an HRESULT error code
        if amsi_hres != S_OK:
            print(f"[!]\tAmsiScanBuffer Error: {amsi_hres}")
            sys.exit()  # Exit if scan fails

        return amsi_res

    # > ---------------------------------------------------------------------------------
    #   Determines if the result of a scan indicates that the content should be blocked
    # > ---------------------------------------------------------------------------------
    """
        void AmsiResultIsMalware(
            r
        );
    """

    def amsi_result_is_malware(self, amsi_res):
        # List of potential detection result codes
        AMSI_RESULTS = {
            AMSI_RESULT.AMSI_RESULT_BLOCKED_BY_ADMIN_END: 'BLOCKED_BY_ADMIN_END',
            AMSI_RESULT.AMSI_RESULT_BLOCKED_BY_ADMIN_START: 'BLOCKED_BY_ADMIN_START',
            AMSI_RESULT.AMSI_RESULT_DETECTED: 'DETECTED'
        }

        if amsi_res.value in AMSI_RESULTS.keys():
            return True

        else:
            return False


if __name__ == '__main__':
    amsi = Amsi()
    amsi.amsi_open_session()
    print(amsi.amsi_scan_string('X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'))
