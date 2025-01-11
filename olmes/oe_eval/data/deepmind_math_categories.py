DEEPMIND_MATH_CATEGORIES = [
    "algebra__linear_1d",
    "algebra__linear_1d_composed",
    "algebra__linear_2d",
    "algebra__linear_2d_composed",
    "algebra__polynomial_roots",
    "algebra__polynomial_roots_composed",
    "algebra__sequence_next_term",
    "algebra__sequence_nth_term",
    "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_in_base",
    "arithmetic__add_sub_multiple",
    "arithmetic__div",
    "arithmetic__mixed",
    "arithmetic__mul",
    "arithmetic__mul_div_multiple",
    "arithmetic__nearest_integer_root",
    "arithmetic__simplify_surd",
    "calculus__differentiate",
    "calculus__differentiate_composed",
    "comparison__closest",
    "comparison__closest_composed",
    "comparison__kth_biggest",
    "comparison__kth_biggest_composed",
    "comparison__pair",
    "comparison__pair_composed",
    "comparison__sort",
    "comparison__sort_composed",
    "measurement__conversion",
    "measurement__time",
    "numbers__base_conversion",
    "numbers__div_remainder",
    "numbers__div_remainder_composed",
    "numbers__gcd",
    "numbers__gcd_composed",
    "numbers__is_factor",
    "numbers__is_factor_composed",
    "numbers__is_prime",
    "numbers__is_prime_composed",
    "numbers__lcm",
    "numbers__lcm_composed",
    "numbers__list_prime_factors",
    "numbers__list_prime_factors_composed",
    "numbers__place_value",
    "numbers__place_value_composed",
    "numbers__round_number",
    "numbers__round_number_composed",
    "polynomials__add",
    "polynomials__coefficient_named",
    "polynomials__collect",
    "polynomials__compose",
    "polynomials__evaluate",
    "polynomials__evaluate_composed",
    "polynomials__expand",
    "polynomials__simplify_power",
    "probability__swr_p_level_set",
    "probability__swr_p_sequence",
]

# Manually selected 3 from 5 randomly chosen examples from the training set corresponding to the task type.
DEEPMIND_MATH_ANSWER_EXAMPLES_PICKED = {
    "algebra__linear_1d": ["4", "68", "-8"],
    "algebra__linear_1d_composed": ["-1", "2", "-5"],
    "algebra__linear_2d": ["-1", "0", "-7"],
    "algebra__linear_2d_composed": ["3", "5", "2"],
    "algebra__polynomial_roots": [
        "1, 5, 156",
        "-5/4, 0, 3/8, 1",
        "-2*(h - 2)*(h + 203426)**2/3",
    ],
    "algebra__polynomial_roots_composed": [
        "4*(g - 2)**5",
        "1, 2",
        "2*(g - 3)*(g - 1)*(g + 1)**2/3",
    ],
    "algebra__sequence_next_term": ["120998303", "110", "-1143"],
    "algebra__sequence_nth_term": [
        "-1993*o + 6",
        "2*k**3 + 2*k**2 + 615*k - 72",
        "-b**3 - 2*b**2 - 6",
    ],
    "arithmetic__add_or_sub": [
        "67570686",
        "15237.9144",
        "-9021",
    ],
    "arithmetic__add_or_sub_in_base": [
        "645",
        "-1a58071",
        "-a037bbb",
    ],
    "arithmetic__add_sub_multiple": ["53", "0", "-27"],
    "arithmetic__div": ["-215738968", "256", "35/3"],
    "arithmetic__mixed": ["-106", "6/5", "-2/7"],
    "arithmetic__mul": ["109953701", "202.4", "461.950704"],
    "arithmetic__mul_div_multiple": ["-2", "-5/3", "2/21"],
    "arithmetic__nearest_integer_root": ["10", "64", "52"],
    "arithmetic__simplify_surd": [
        "-2580 + 400*sqrt(5)",
        "-5*sqrt(13)",
        "-4 + sqrt(2)",
    ],
    "calculus__differentiate": [
        "19604320",
        "17381146*t - 2",
        "-648*k*l*q**2 + 6*k*q - 2832*l",
    ],
    "calculus__differentiate_composed": [
        "-23520",
        "-48*t",
        "-60*m**3 - 204*m**2",
    ],
    "comparison__closest": ["c", "-0.138", "a"],
    "comparison__closest_composed": ["2", "-0.2", "a"],
    "comparison__kth_biggest": ["f", "-3", "g"],
    "comparison__kth_biggest_composed": ["p", "2", "0"],
    "comparison__pair": ["False", "-1", "2894"],
    "comparison__pair_composed": ["-5/117", "t", "False"],
    "comparison__sort": [
        "55, 3, -3, -5, -29",
        "37, 1/7, 0, -0.1, -2",
        "-6, -47, -108",
    ],
    "comparison__sort_composed": [
        "0.4, z, -2",
        "j, -3, w",
        "-2/3, g, h",
    ],
    "measurement__conversion": [
        "200",
        "50.55786",
        "0.000539861875",
    ],
    "measurement__time": ["1:58 AM", "6:38 PM", "588"],
    "numbers__base_conversion": ["-1", "223000121", "-772779"],
    "numbers__div_remainder": ["52", "11", "7"],
    "numbers__div_remainder_composed": ["17", "8", "5"],
    "numbers__gcd": ["1", "1601", "15"],
    "numbers__gcd_composed": ["1", "26", "5"],
    "numbers__is_factor": ["False", "True"],
    "numbers__is_factor_composed": ["True", "False"],
    "numbers__is_prime": ["True", "False"],
    "numbers__is_prime_composed": ["False", "True"],
    "numbers__lcm": ["28004262", "10860", "5759271"],
    "numbers__lcm_composed": ["4", "8396", "27"],
    "numbers__list_prime_factors": [
        "3, 17, 71",
        "3, 301813",
        "47, 643",
    ],
    "numbers__list_prime_factors_composed": [
        "3, 79",
        "757",
        "2, 7, 11",
    ],
    "numbers__place_value": ["1", "8", "5"],
    "numbers__place_value_composed": ["5", "2", "7"],
    "numbers__round_number": ["0.1", "122620000", "-0.888"],
    "numbers__round_number_composed": [
        "130000",
        "0.00043",
        "0.0053",
    ],
    "polynomials__add": [
        "2*n + 2",
        "3*c**3 + 339*c - 2",
        "-22*a**3 + 1",
    ],
    "polynomials__coefficient_named": ["-3", "533", "-250"],
    "polynomials__collect": ["-3*p**2 + 416", "k", "-8*c"],
    "polynomials__compose": [
        "-578*g**4",
        "-8*s**2 + 2",
        "x**2",
    ],
    "polynomials__evaluate": ["-2477", "-95", "23"],
    "polynomials__evaluate_composed": ["5", "24", "-1"],
    "polynomials__expand": [
        "7*r**5 + 7*r**4",
        "60*u**2 + 4*u",
        "-11*q**4",
    ],
    "polynomials__simplify_power": [
        "t**(-594/5)",
        "x**(4895/399)",
        "d**(-228)",
    ],
    "probability__swr_p_level_set": ["1/35", "1/120", "35/51"],
    "probability__swr_p_sequence": ["1/840", "14/323", "1/60"],
}