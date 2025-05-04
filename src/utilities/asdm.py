import numpy as np
import scipy.special
import scipy.signal
import logging

logger = logging.getLogger(__name__)


def asdm_encode(u: np.ndarray, dt: float, d_norm: float, dte: float = 0.0, y: float = 0.0, interval: float = 0.0, sgn: int = 1, quad_method: str = 'trapz') -> np.ndarray:

    '''
    ASDM time encoding machine.

    Encode a finite length signal using an Asynchronous Sigma-Delta Modulator.

    Inputs
    ----------
    - u: array_like of floats
        Signal to encode.
    - dt: float
        Sampling resolution of input signal; the sampling frequency is 1/dt Hz.
    - d_norm: float
        Normalized hysteresis threshold. The encoder will generate a spike
        whenever the value of the integrator exceeds this threshold.
    - dte: float
        Sampling resolution assumed by the encoder (s). This may not exceed `dt`.
    - y: float
        Initial value of integrator.
    - interval: float
        Time since last spike (in s).
    - sgn: int -> {+1, -1}
        Sign of integrator.
    - quad_method: {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).

    Outputs
    -------
    - s: ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    '''
    Nu = len(u)
    if Nu == 0:
       return np.array([], np.float) 

    if dte > dt:
        raise ValueError('Encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('Encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:
        # Resample signal and adjust signal length accordingly:
        M = int(dt/dte)
        u = scipy.signal.resample(u, len(u)*M)
        Nu *= M
        dt = dte

    # Choose integration method and set the number of points over which to integrate the input (see note above). 
    # This allows the use of one loop below to perform the integration regardless of the method chosen:
    s = []
    if quad_method == 'rect':
        compute_y = lambda y, sgn, i: y + dt*(sgn + u[i])
        last = Nu
    elif quad_method == 'trapz':
        compute_y = lambda y, sgn, i : y + dt*(sgn+(u[i]+u[i+1])/2.0)
        last = Nu-1
    else:
        raise ValueError('unrecognized quadrature method')

    for i in range(last):
        y = compute_y(y, sgn, i)
        interval += dt
        if np.abs(y) >= d_norm:
            s.append(interval)
            interval = 0.0
            y = d_norm * sgn
            sgn = -sgn

    return np.array(s)


def asdm_decode(s: np.ndarray, dur: float, dt: float, bw: float, sgn: int = -1) -> np.ndarray:
    '''
    Threshold-insensitive ASDM time decoding machine.

    Decode a signal encoded with an Asynchronous Sigma-Delta Modulator using a threshold-insensitive recovery algorithm.

    Inputs
    ----------
    - s: ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).
    - dur: float
        Duration of signal (in s).
    - dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    - bw: float
        Signal bandwidth (in rad/s).
    - b: float
        Encoder bias.
    - sgn: int -> {-1, 1}
        Sign of first spike.

    Outputs
    -------
    - u_rec: ndarray of floats
        Recovered signal.
    '''
    __pinv_rcond__ = 1e-15  # Constant for pseudoinverse stability

    Ns = len(s)
    if Ns < 2:
        raise ValueError('Spike train must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s) 
    # Convert spike intervals to spike times
    ts = np.cumsum(s) 
    # Compute the midpoints between spike times --> sinc interpolation points
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    t = np.arange(0, dur, dt)
    bwpi = bw/np.pi

    # --- Build Reconstruction Matrices ---
    G = np.empty((Nsh, Nsh), dtype=float)
    try: 
        for j in range(Nsh):
            temp = scipy.special.sici(bw*(ts-tsh[j]))[0]/np.pi
            G[:, j] = temp[1:]-temp[:-1]
    except Exception as e:
        logger.error(f"Error calculating Si function or G matrix elements: {e}", exc_info=True)
        raise RuntimeError("Failed to build G matrix for decoding.") from e

    # Apply compensation principle:
    B = np.diag(np.ones(Nsh - 1), -1) + np.eye(Nsh)
    if sgn == -1:
        Bq = (-1)**np.arange(Nsh) * (s[1:] - s[:-1])
    else:
        Bq = (-1)**np.arange(1, Nsh + 1) * (s[1:] - s[:-1])

    # --- Solve for Coefficients ---
    # Reconstruct signal by adding up the weighted sinc functions; the first row of B is removed to eliminate boundary issues. 
    # The weighted sinc functions are computed on the fly to save memory:
    u_rec = np.zeros(len(t), dtype=float)
    try: 
        c = np.dot(np.linalg.pinv(np.dot(B[1:, :], G), __pinv_rcond__), Bq[1:, np.newaxis])
        for i in range(Nsh):
            u_rec += np.sinc(bwpi * (t - tsh[i])) * bwpi * c[i]
        return u_rec
    except Exception as e:
        logger.error(f"Error during final signal reconstruction loop: {e}", exc_info=True)
        raise RuntimeError("Failed during sinc interpolation sum.") from e
    

def asdm_encode_physical(u: np.ndarray, dt: float, k: float, b: float, delta: float, dte: float = 0.0, y: float = 0.0, interval: float = 0.0, sgn: int = 1, quad_method: str = 'trapz') -> np.ndarray:

    '''
    ASDM time encoding machine.

    Encode a finite length signal using an Asynchronous Sigma-Delta Modulator.

    Inputs
    ----------
    - u: array_like of floats
        Signal to encode.
    - dt: float
        Sampling resolution of input signal; the sampling frequency is 1/dt Hz.
    - k: float
        Encoder gain (in s).
    - b: float
        Encoder bias (in s).
    - delta: float
        Encoder hysteresis threshold (in s).
    - dte: float
        Sampling resolution assumed by the encoder (s). This may not exceed `dt`.
    - y: float
        Initial value of integrator.
    - interval: float
        Time since last spike (in s).
    - sgn: int -> {+1, -1}
        Sign of integrator.
    - quad_method: {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).

    Outputs
    -------
    - s: ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    '''
    Nu = len(u)
    if Nu == 0:
       return np.array([], np.float) 

    if dte > dt:
        raise ValueError('Encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('Encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:
        # Resample signal and adjust signal length accordingly:
        M = int(dt/dte)
        u = scipy.signal.resample(u, len(u)*M)
        Nu *= M
        dt = dte

    # Choose integration method and set the number of points over which to integrate the input (see note above). 
    # This allows the use of one loop below to perform the integration regardless of the method chosen:
    s = []
    if quad_method == 'rect':
        compute_y = lambda y, sgn, i: y + dt*(sgn*b+u[i])/k
        last = Nu
    elif quad_method == 'trapz':
        compute_y = lambda y, sgn, i : y + dt*(sgn*b+(u[i]+u[i+1])/2.0)/k
        last = Nu-1
    else:
        raise ValueError('unrecognized quadrature method')

    for i in range(last):
        y = compute_y(y, sgn, i)
        interval += dt
        if np.abs(y) >= delta:
            s.append(interval)
            interval = 0.0
            y = delta*sgn
            sgn = -sgn

    return np.array(s)


def asdm_decode_physical(s: np.ndarray, dur: float, dt: float, b: float, bw: float, sgn: int = -1) -> np.ndarray:
    '''
    Threshold-insensitive ASDM time decoding machine.

    Decode a signal encoded with an Asynchronous Sigma-Delta Modulator using a threshold-insensitive recovery algorithm.

    Inputs
    ----------
    - s: ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).
    - dur: float
        Duration of signal (in s).
    - dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    - b: float
        Encoder bias (in s).
    - bw: float
        Signal bandwidth (in rad/s).
    - b: float
        Encoder bias.
    - sgn: int -> {-1, 1}
        Sign of first spike.

    Outputs
    -------
    - u_rec: ndarray of floats
        Recovered signal.
    '''
    __pinv_rcond__ = 1e-15  # Constant for pseudoinverse stability

    Ns = len(s)
    if Ns < 2:
        raise ValueError('Spike train must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s) 
    # Convert spike intervals to spike times
    ts = np.cumsum(s) 
    # Compute the midpoints between spike times --> sinc interpolation points
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    t = np.arange(0, dur, dt)
    bwpi = bw/np.pi

    # --- Build Reconstruction Matrices ---
    G = np.empty((Nsh, Nsh), dtype=float)
    try: 
        for j in range(Nsh):
            temp = scipy.special.sici(bw*(ts-tsh[j]))[0]/np.pi
            G[:, j] = temp[1:]-temp[:-1]
    except Exception as e:
        logger.error(f"Error calculating Si function or G matrix elements: {e}", exc_info=True)
        raise RuntimeError("Failed to build G matrix for decoding.") from e

    # Apply compensation principle:
    B = np.diag(np.ones(Nsh-1), -1)+np.eye(Nsh)
    if sgn == -1:
        Bq = (-1)**np.arange(Nsh)*b*(s[1:]-s[:-1])
    else:
        Bq = (-1)**np.arange(1, Nsh+1)*b*(s[1:]-s[:-1])

    # --- Solve for Coefficients ---
    # Reconstruct signal by adding up the weighted sinc functions; the first row of B is removed to eliminate boundary issues. 
    # The weighted sinc functions are computed on the fly to save memory:
    u_rec = np.zeros(len(t), dtype=float)
    try: 
        c = np.dot(np.linalg.pinv(np.dot(B[1:, :], G), __pinv_rcond__), Bq[1:, np.newaxis])
        for i in range(Nsh):
            u_rec += np.sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
        return u_rec
    except Exception as e:
        logger.error(f"Error during final signal reconstruction loop: {e}", exc_info=True)
        raise RuntimeError("Failed during sinc interpolation sum.") from e