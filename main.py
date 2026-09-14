import argparse

from src.analysis import check_conservation, print_conservation_report
from src.integrator import integrator
from src.scenarios import SCENARIOS, find_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Моделирование столкновения двух гладких шаров"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=SCENARIOS[0].name,
        help="Название сценария",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать названия доступных сценариев",
    )
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--slowdown", type=float, default=100.0)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    if args.list:
        for scenario in SCENARIOS:
            print(scenario.name)
        return

    scenario = find_scenario(args.scenario)
    if scenario is None:
        parser.error(f"Сценарий {args.scenario!r} не найден. Используйте --list")

    from src.visualization import animate_simulation

    t, y = integrator(
        scenario.ball1,
        scenario.ball2,
        scenario.k,
        scenario.t_span,
    )
    conservation = check_conservation(
        y,
        scenario.ball1,
        scenario.ball2,
        scenario.k,
        rtol=scenario.rtol,
        atol=scenario.atol,
    )
    print(f"Сценарий: {scenario.name}")
    print_conservation_report(conservation)
    animation = animate_simulation(
        t,
        y,
        scenario.ball1,
        scenario.ball2,
        conservation,
        fps=args.fps,
        playback_speed=args.speed,
        collision_slowdown=args.slowdown,
    )


if __name__ == "__main__":
    main()
