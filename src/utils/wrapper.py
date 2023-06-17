import typing
from typing import Any, Hashable, Iterable, Sequence
import jax

T = typing.TypeVar("T", bound=typing.Callable[..., Any])


def typed_jit(
    fun: T,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] = (),
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> T:
    return typing.cast(
        T,
        jax.jit(
            fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            abstracted_axes=abstracted_axes,
        ),
    )


def typed_grad(
    fun: T,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[Hashable] = (),
) -> T:
    return typing.cast(
        T,
        jax.grad(
            fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        ),
    )
