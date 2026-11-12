import lab4


def run_additional_tests():
    """Запускает проверку полным перебором (п. 8.2) и 6 граничных тестов (п. 8.3)."""
    print("\n" + "=" * 85)
    print("Результаты дополнительных тестов и проверки полным перебором")
    print("=" * 85)

    # Сохраняем исходные константы модуля lab4
    orig = (
        lab4.START_ASSETS, lab4.START_CASH, lab4.MIN_LIMITS,
        lab4.PACK_SIZES, lab4.BROKER_FEES, lab4.MARKET_STAGES, lab4.ENABLE_PRUNING
    )

    def restore():
        (lab4.START_ASSETS, lab4.START_CASH, lab4.MIN_LIMITS,
         lab4.PACK_SIZES, lab4.BROKER_FEES, lab4.MARKET_STAGES, lab4.ENABLE_PRUNING) = orig
        lab4.solve_bellman.cache_clear()

    try:
        # Тест 1: Нулевой портфель и нулевой остаток денег
        st_zero = (0.0, 0.0, 0.0, 0.0)
        acts_1 = [u for _, u in lab4.generate_valid_trades(st_zero)]
        assert acts_1 == [(0, 0, 0)], f"Ошибка теста 1: {acts_1}"
        print(f"1. Нулевой портфель s=(0,0,0,0):                 ожид. [(0, 0, 0)] -> факт. {acts_1} [ПРОЙДЕН]")

        # Тест 2: Один доступный для покупки доходный актив
        lab4.PACK_SIZES = (4.0, 16.0, 16.0)
        lab4.BROKER_FEES = (0.25, 0.25, 0.25)
        lab4.MARKET_STAGES = (
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (2.0, 1.0, 1.0)),),
        )
        lab4.solve_bellman.cache_clear()
        v_2, ch_2 = lab4.evaluate_last_stage((0.0, 0.0, 0.0, 5.0))
        u_2, post_2 = ch_2[0]
        assert u_2 == (1, 0, 0) and abs(v_2 - 8.0) < lab4.TOLERANCE
        print(f"2. Один доступный актив (C=5, пакет=4, ком=25%):  ожид. (1, 0, 0), V=8.0 -> факт. {u_2}, V={v_2:.1f} [ПРОЙДЕН]")

        # Тест 3: Весь бюджет направляется в один актив (C' = 0)
        assert abs(post_2[3] - 0.0) < lab4.TOLERANCE
        print(f"3. Весь бюджет одному активу (остаток C'=0):      ожид. C'=0.0, V=8.0 -> факт. C'={post_2[3]:.1f}, V={v_2:.1f} [ПРОЙДЕН]")
        restore()

        # Тест 4: Нулевой рост рынка при ненулевых комиссиях (оптимально бездействие)
        lab4.MARKET_STAGES = (
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (1.0, 1.0, 1.0)),),
        )
        lab4.solve_bellman.cache_clear()
        start_st = (100.0, 800.0, 400.0, 600.0)
        v_4, ch_4 = lab4.solve_bellman(0, start_st)
        u_4 = [u for u, _ in ch_4]
        assert abs(v_4 - 1900.0) < lab4.TOLERANCE and u_4 == [(0, 0, 0)]
        print(f"4. Нулевая доходность (с комиссиями):             ожид. V=1900.0, [(0, 0, 0)] -> факт. V={v_4:.1f}, {u_4} [ПРОЙДЕН]")

        # Тест 5: Несколько равнозначных оптимумов (нулевые комиссии и нулевой рост)
        lab4.BROKER_FEES = (0.0, 0.0, 0.0)
        lab4.solve_bellman.cache_clear()
        v_5, ch_5 = lab4.solve_bellman(0, start_st)
        assert abs(v_5 - 1900.0) < lab4.TOLERANCE and len(ch_5) > 1
        print(f"5. Несколько оптимумов (нулевые комиссии):        ожид. >1 плана, V=1900.0 -> факт. {len(ch_5)} планов, V={v_5:.1f} [ПРОЙДЕН]")
        restore()

        # Тест 6: Некорректные входные данные (отрицательный кэш и сумма вероятностей != 1)
        caught_errors = 0
        lab4.START_CASH = -50.0
        try:
            lab4.check_parameters()
        except ValueError:
            caught_errors += 1
        restore()

        lab4.MARKET_STAGES = (
            ((0.50, (1.2, 1.1, 1.0)), (0.30, (1.0, 1.0, 1.0))),
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (1.0, 1.0, 1.0)),),
        )
        try:
            lab4.check_parameters()
        except ValueError:
            caught_errors += 1
        assert caught_errors == 2
        print(f"6. Некорректные входные данные (2 проверки):      ожид. 2 ValueError -> факт. {caught_errors} ValueError [ПРОЙДЕН]")
        restore()

        # Тест 7: Независимая проверка полным перебором на малом примере (п. 8.2)
        lab4.START_ASSETS = (4.0, 0.0, 0.0)
        lab4.START_CASH = 0.0
        lab4.MIN_LIMITS = (0.0, 0.0, 0.0)
        lab4.PACK_SIZES = (4.0, 16.0, 16.0)
        lab4.BROKER_FEES = (0.25, 0.25, 0.25)
        lab4.MARKET_STAGES = (
            ((0.5, (2.0, 1.0, 1.0)), (0.5, (0.5, 1.0, 1.0))),
            ((1.0, (1.0, 1.0, 1.0)),),
            ((1.0, (1.0, 1.0, 1.0)),),
        )
        lab4.solve_bellman.cache_clear()

        def brute_force(stage, state):
            if stage == 3:
                return lab4.calc_total_capital(state)
            best = float("-inf")
            for post_tr, _ in lab4.generate_valid_trades(state):
                exp_v = sum(
                    p * brute_force(stage + 1, lab4.apply_market_step(post_tr, m))
                    for p, m in lab4.MARKET_STAGES[stage]
                )
                if exp_v > best:
                    best = exp_v
            return best

        small_st = (4.0, 0.0, 0.0, 0.0)
        dp_val, _ = lab4.solve_bellman(0, small_st)
        bf_val = brute_force(0, small_st)
        assert abs(dp_val - bf_val) < lab4.TOLERANCE and abs(dp_val - 5.0) < lab4.TOLERANCE
        print(f"7. Проверка полным перебором (малый пример):      ожид. DP == Перебор == 5.0 -> факт. {dp_val:.2f} == {bf_val:.2f} [ПРОЙДЕН]")
    finally:
        restore()

    print("=" * 85)


if __name__ == "__main__":
    run_additional_tests()
