from collections import namedtuple


def format(number):
	return "{:,}".format(round(number, 0))


def gen(new_amnt, interest):
	t = 0
	Account = namedtuple('Account', 'new profit  monthly_profit t')
	while True:
		t += 1
		new_amnt, old_amnt = new_amnt * (1 + interest) ** t, new_amnt
		profit = new_amnt-old_amnt
		yield Account(format(new_amnt), format(profit), format(profit/12), t)


gen_ = gen(100_000, 0.04) # what we have on bank account, interest rate by idea
# gen_=gen(100,0.1)

for _ in range(15):
	print(next(gen_))


# =============================================================================
# Debugger
# =============================================================================