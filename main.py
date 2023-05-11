from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate the moon-shaped dataset
moonsX, moonsY = make_moons(n_samples=1000, noise=0.3, random_state=42)
digitsX, digitsY = load_digits(n_samples=1000, noise=0.3, random_state=42)
# Split the data into train and test sets
moonsX_train, moonsX_test, moonsY_train, moonsY_test = train_test_split(moonsX, moonsY, test_size=0.2, random_state=42)
digitsX_train, digitsX_test, digitsY_train, digitsY_test = train_test_split(digitsX, digitsY, test_size=0.2, random_state=42)

simple = LogisticRegression()
simple.fit(moonsX_train, moonsY_train)
simple2 = LogisticRegression()
simple2.fit(digitsX_train, digitsY_train)

multi = LogisticRegression(multi_class="multinomial", solver="lbfgs")
multi.fit(moonsX_train, moonsY_train)
multi2 = LogisticRegression(multi_class="multinomial", solver="lbfgs")
multi2.fit(digitsX_train, digitsY_train)

reg = LogisticRegression(penalty="l2")
reg.fit(moonsX_train, moonsY_train)
reg2 = LogisticRegression(penalty="l2")
reg2.fit(digitsX_train, digitsY_train)

reg_multi = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2")
reg_multi.fit(moonsX_train, moonsY_train)
reg_multi2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2")
reg_multi2.fit(digitsX_train, digitsY_train)

pp_simple = simple.predict_proba(moonsX_test)
pp_simple2 = simple2.predict_proba(digitsX_test)

pp_multi = multi.predict_proba(moonsX_test)
pp_multi2 = multi2.predict_proba(digitsX_test)

pp_reg = reg.predict_proba(moonsX_test)
pp_reg2 = reg2.predict_proba(digitsX_test)

pp_reg_multi = reg_multi.predict_proba(moonsX_test)
pp_reg_multi2 = reg_multi2.predict_proba(digitsX_test)

simple_pred = simple.predict(moonsX_test)
simple2_pred = simple2.predict(digitsX_test)

multi_pred = multi.predict(moonsX_test)
multi2_pred = multi2.predict(digitsX_test)

reg_pred = reg.predict(moonsX_test)
reg2_pred = reg2.predict(digitsX_test)

reg_multi_pred = reg_multi.predict(moonsX_test)
reg_multi2_pred = reg_multi2.predict(digitsX_test)