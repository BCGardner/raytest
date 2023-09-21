# test.py
import ray


# `auto` is passed to allow the head node
# to determine the networking.
ray.init(address='auto')


# Functions can be decorated to tell Ray what function
# will be distributed for compute.
# Decorators work perfectly for simple functions.
@ray.remote
def f(x):
    return x * x


# Manual data processing is done to collect results.
futures = [f.remote(i) for i in range(200)]
results = ray.get(futures)
print(results)

print(ray.__version__)
