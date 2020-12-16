FROM tomyun/cropbox:latest

COPY --chown=${USER} . ${HOME}

RUN JULIA_PROJECT=${HOME} julia -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
