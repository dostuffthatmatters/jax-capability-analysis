from typing import Any
import numpy
import scipy.optimize

from src import utils


def run(
    data_from_sensor_a: numpy.ndarray[numpy.float32, Any],
    data_from_sensor_b: numpy.ndarray[numpy.float32, Any],
    actual_slope: float,
    actual_offset: float,
) -> None:
    # define the loss function

    def rmse_with_calibration(x: numpy.ndarray[float, Any]) -> Any:
        return numpy.sqrt(
            numpy.mean(
                numpy.power(
                    data_from_sensor_a - ((data_from_sensor_b - x[1]) / x[0]),
                    2,
                )
            )
        )

    # running the rmse calculation 10 times

    for i in range(5):
        with utils.timing.timed_section(f"run rmse_with_calibration (iteration #{i})"):
            rmse_with_calibration(numpy.array([9.0, 7.0]))

    # minimize the rmse with respect to the slope and offset

    initial_guess = numpy.array([1.0, 0.0])  # (slope, offset)
    optimization_result = scipy.optimize.minimize(
        rmse_with_calibration,
        x0=initial_guess,
        tol=0.0001,
        method="BFGS",
        options={"disp": True},
    )
    final_guess = optimization_result.x
    print("optimization_result:", optimization_result)

    utils.assertions.assert_similar_result(
        "optimized slope", actual_slope, final_guess[0], precision=1
    )
    utils.assertions.assert_similar_result(
        "optimized offset", actual_offset, final_guess[1], precision=1
    )
