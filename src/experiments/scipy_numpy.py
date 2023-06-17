from typing import Any
import numpy
import scipy.optimize

from src import utils

# TODO: plot input and output with plotly and save it to ./out


def run(
    data_from_sensor_a: numpy.ndarray[numpy.float32, Any],
    data_from_sensor_b: numpy.ndarray[numpy.float32, Any],
    actual_slope: float,
    actual_offset: float,
) -> None:
    # define the loss function

    def rmse_with_calibration(x: tuple[float, float]) -> float:
        estimated_slope, estimated_offset = x
        return float(
            numpy.sqrt(
                numpy.mean(
                    numpy.power(
                        data_from_sensor_a
                        - (estimated_slope * data_from_sensor_b + estimated_offset),
                        2,
                    )
                )
            )
        )

    # running the rmse calculation 10 times

    for i in range(10):
        with utils.timing.timed_section(f"rmse_with_calibration run #{i}"):
            rmse_with_calibration((9, 7))

    # minimize the rmse with respect to the slope and offset

    initial_guess = (1, 0)  # (slope, offset)
    optimization_result = scipy.optimize.minimize(
        rmse_with_calibration,
        x0=initial_guess,
        method="BFGS",
    )
    final_guess = optimization_result.x
    print("optimization_result:", optimization_result)

    # expected result: slope = 1.5, offset = 3

    utils.assertions.assert_similar_result(
        "optimized slope", final_guess[0], actual_slope
    )
    utils.assertions.assert_similar_result(
        "optimized offset", final_guess[1], actual_offset
    )
