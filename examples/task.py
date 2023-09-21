import ray


def main():
    @ray.remote
    def f(x):
        return x * x

    futures = [f.remote(i) for i in range(4)]
    print(ray.get(futures)) # [0, 1, 4, 9]


if __name__ == '__main__':
    ray.init()
    main()
