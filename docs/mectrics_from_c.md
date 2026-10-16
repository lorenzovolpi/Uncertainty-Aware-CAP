With the convention

$$
C_{rk}=P(\widehat Y=r\mid Y=k),
\qquad
T=C\operatorname{diag}(\pi),
$$

the normalized contingency-table entry is

$$
T_{rk}=C_{rk}\pi_k.
$$

## vanilla accuracy

$$
\boxed{
\operatorname{Acc}(C,\pi)=\operatorname{diag}(C)^\top\pi.
}
$$

## macro-F1

For class $k$:

$$
\mathrm{TP}_k=C_{kk}\pi_k,
$$

$$
\mathrm{FP}_k=(C\pi)_k-C_{kk}\pi_k,
$$

$$
\mathrm{FN}_k=\pi_k-C_{kk}\pi_k.
$$

Thus its class-wise F1 is

$$
\mathrm{F1}_k(C,\pi)
=
\frac{2\,C_{kk}\pi_k}
{2\,C_{kk}\pi_k+\mathrm{FP}_k+\mathrm{FN}_k}
=
\frac{2\,C_{kk}\pi_k}
{(C\pi)_k+\pi_k}.
$$

$(C\pi)_k$ is the target marginal prevalence of predictions assigned to class $k$:

$$
(C\pi)_k
=
P(\widehat Y=k)
=
\sum_{j=1}^K C_{kj}\pi_j
=
\sum_{j=1}^K
P(\widehat Y=k\mid Y=j)\,P(Y=j).
$$

It is the $k$-th row sum of the normalized contingency table:

$$
(C\pi)_k
=
\sum_{j=1}^K T_{kj}.
$$

For macro-F1, it is the predicted-positive mass for class $k$, i.e. the sum of true positives and false positives:

$$
(C\pi)_k
=
\mathrm{TP}_k+\mathrm{FP}_k.
$$

If we define

$$
\mathbf{c}=C\pi,
$$

then

$$
c_k=(C\pi)_k=P(\widehat Y=k).
$$

Therefore,

$$
\boxed{
\operatorname{MacroF1}(C,\pi)
=
\frac{1}{K}
\sum_{k=1}^K
\frac{2\,C_{kk}\pi_k}
{c_k+\pi_k}
}
$$

with an explicit convention required when the denominator is zero, i.e. when class $k$ is absent both from the true labels and from the predictions. This is nonlinear in $\pi$.

Here:

$$
C_{kk}\pi_k=\mathrm{TP}_k,
\qquad
c_k=\mathrm{TP}_k+\mathrm{FP}_k,
\qquad
\pi_k=\mathrm{TP}_k+\mathrm{FN}_k.
$$

## micro-F1

For the single-label multiclass setting of the paper, micro-F1 is exactly vanilla accuracy.

Let

$$
\mathrm{TP}_{\mathrm{micro}}
=
\sum_{k=1}^K \mathrm{TP}_k
=
\sum_{k=1}^K C_{kk}\pi_k.
$$

Since every instance has exactly one true class and exactly one predicted class,

$$
\mathrm{FP}_{\mathrm{micro}}
=
\mathrm{FN}_{\mathrm{micro}}
=
1-\mathrm{TP}_{\mathrm{micro}}.
$$

Thus,

$$
\operatorname{MicroF1}(C,\pi)
=
\frac{
2\mathrm{TP}_{\mathrm{micro}}
}{
2\mathrm{TP}_{\mathrm{micro}}
+
\mathrm{FP}_{\mathrm{micro}}
+
\mathrm{FN}_{\mathrm{micro}}
}.
$$

Substituting gives

$$
\operatorname{MicroF1}(C,\pi)
=
\frac{
2\sum_{k=1}^K C_{kk}\pi_k
}{
2
}
=
\sum_{k=1}^K C_{kk}\pi_k.
$$

Therefore,

$$
\boxed{
\operatorname{MicroF1}(C,\pi)
=
\operatorname{Acc}(C,\pi)
=
\operatorname{diag}(C)^\top\pi
}
$$

This equivalence holds for standard single-label classification; it does not generally hold in multilabel classification.