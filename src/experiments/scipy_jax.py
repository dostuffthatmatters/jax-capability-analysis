from typing import Any
import numpy
import jax.numpy
import jax.scipy.optimize

from src import utils

# TODO: plot input and output with plotly and save it to ./out


def run(
    data_from_sensor_a: numpy.ndarray[numpy.float32, Any],
    data_from_sensor_b: numpy.ndarray[numpy.float32, Any],
    actual_slope: float,
    actual_offset: float,
) -> None:
    # define the loss function

    data_from_sensor_a_jax = jax.numpy.array(data_from_sensor_a)
    data_from_sensor_b_jax = jax.numpy.array(data_from_sensor_b)

    @jax.jit
    def rmse_with_calibration(x: jax.Array) -> jax.Array:
        estimated_slope = x[0]
        estimated_offset = x[1]
        return jax.numpy.sqrt(
            jax.numpy.mean(
                jax.numpy.power(
                    data_from_sensor_a_jax
                    - (estimated_slope * data_from_sensor_b_jax + estimated_offset),
                    2,
                )
            )
        )

    # running the rmse calculation 10 times

    for i in range(10):
        with utils.timing.timed_section(f"rmse_with_calibration run #{i}"):
            rmse_with_calibration((9, 7))

    # minimize the rmse with respect to the slope and offset

    initial_guess = jax.numpy.array([1.0, 0.0])  # (slope, offset)
    optimization_result = jax.scipy.optimize.minimize(
        rmse_with_calibration,
        x0=initial_guess,
        method="BFGS",
    )
    final_guess: Any = optimization_result.x
    print("optimization_result:", optimization_result)

    # expected result: slope = 1.5, offset = 3

    utils.assertions.assert_similar_result(
        "optimized slope", final_guess[0], actual_slope
    )
    utils.assertions.assert_similar_result(
        "optimized offset", final_guess[1], actual_offset
    )
