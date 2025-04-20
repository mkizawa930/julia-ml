
# ベイズ推定


## エビデンス下界(ELBO)

観測変数をX, 事前分布のような潜在変数をZ, ハイパーパラメータを$\theta$とすると、対数周辺尤度(モデルエビデンス) $p_{\theta}(x)$ はZについて周辺化すると得られる。
$$
\log p_{\theta}(x) = \log \int p_{\theta}(x, z) dz
$$

頻度統計においては尤度関数のパラメータについて最尤推定するのに対して、ベイズ統計では事前分布が現れるので観測データに対してモデルを評価するには観測変数とパラメータの関数にした、この周辺尤度を用いてモデルを定量的に評価を行う。(モデルのデータへの当てはまりを表す)


まず、ベイズの定理から次の式が成り立つ
$$
p(x) = \frac{p(x,z)}{p(z|x)}
$$

これを対数周辺尤度に代入すると、

$$
\log p(x | \theta) \\
= \log \int p(x,z | \theta) dz \\
= \log \int \frac{p_{\theta}(x,z)}{q(z)} + {\rm KL}[q(z) || p_{\theta}(z|x)]
\geq \log \int q(z) \frac{p(x,z)}{q(z)} dz + \int q(z) \log \frac{p(x,z)}{q(z)} dz
$$

特に右辺の第二項はKLダイバージェンスで非負なので、次の不等式が成り立つ。

$$
\ln p(x | \theta) \geq {\mathcal L}(q, \theta)
$$

$\mathcal L$は対数周辺尤度の下限であることからELBO(Evidence Lower Bound)と呼ばれる。


## EMアルゴリズム




### Eステップ

$$
q_{new}(z) \leftarrow q_{old}(z)
$$

$\theta_{old}$を固定した状態で、周辺尤度が最大をとる$q(z)$は真の事後分布$p(z|x)$となる。
したがって、求める近似事後分布 $q(z)$ は、

$$
q(z) = p(z | x) = \frac{p(x | z)p(z)}{\int p(x | z) p(z) dz}
$$



### Mステップ

コスト関数をELBOとして

$$
{\rm ELBO}(q, x, \theta)
$$

$$
Q(\theta | \theta_{old})
$$

$$
\theta_{new} = \arg \max_\theta \int q(z) \log \frac{p(x,z)}{q(z)} dz
$$


## 混合ガウス分布におけるベイズ推定


### EMアルゴリズム

モデル

潜在変数Zはクラスタkを割り当てる離散確率分布(カテゴリカル分布)

$$
p(z = k) = \pi_k, \space k=1,...K \\
$$

$$
p(x | z = k) = {\mathcal N}(\boldsymbol \mu_k, \Sigma_k) \\
$$

$$
p(x, z = k) = \pi_k {\mathcal N}(\boldsymbol \mu_k, \Sigma_k), {\rm where}  \space k = 1, ..., K
$$

Xに関する周辺分布は次のように求まる

$$
p(x) = \sum_{k=1}^K \pi_k {\mathcal N}(\boldsymbol \mu_k, \Sigma_k)
$$

EMアルゴリズムでは、近似事後分布$q(z)$とパラメータ$\theta$を交互に更新する

**Eステップ**

混合ガウス分布では真の事後分布$p(z|x)$が求まる．したがって、Eステップでは、$\theta$を固定して、近似事後分布$q(z)$を$p_{\theta_{old}}(z|x)$として求める。

$$
q(z=k) = p_{\theta_{old}}(z|x) = \frac{p(x,z=k)}{\sum p(x,z=k)} = \frac{\pi_k {\mathcal N}(\boldsymbol \mu_k, \Sigma_k)}{\sum \pi_k {\mathcal N}(\boldsymbol \mu_k, \Sigma_k)}
$$


**Mステップ**

ELBOに代入した式を各パラメータについて偏微分して、パラメータについて解いた式に、テップで求めた近似事後分布を代入して、パラメータを更新する。

$$
$$


## 変分EMアルゴリズム

EMアルゴリズムでは潜在変数Zとパラメータ$\theta$を交互に更新することで周辺尤度を最大化した。一方で、パラメータ自体に事前分布を導入したモデルを考えると、真の事後分布は潜在変数とパラメータの混合モデルとなるため、解析的に求まらない場合ことがある。このとき、潜在変数とパラメータを因子化した近似事後分布を考えるとする。

$$
p(z, \theta | x) \approx q(z) q(\theta)
$$

ここで、パラメータの事前分布のパラメータは何らかの初期値を設定しておくとして、近似事後分布を求める方法を考える。
真の事後分布と近似事後分布のKLダイバージェンスは次のように与えられる。

$$
{\rm KL}[q(z) q(\theta) || p(z, \theta | x)] = \int q(z) q(\theta) \log \frac{q(z) q(\theta)}{p(z, \theta | x)} dz d\theta
$$

パラメータの近似事後分布$q(\theta)$が与えられたもとで、KLダイバージェンスを最小化する近似事後分布$q(z)$は

$$
\ln q(z) = {\rm E}_{q(z)} [ \ln p(z, \theta) ]
$$

同様に、$q(z)$が与えられもとでの、近似事後分布$q(\theta)$も期待値計算により求めることができる

$$
\ln q(\theta) = {\rm E}_{q(\theta)} [ \ln p(z, \theta) ]
$$