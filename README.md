# Boosting-AdaBoost

	Background
		Boosting algorithms, which construct a strong (binary) classifier based on iteratively
	adding one weak (binary) classifier into it. A weak classifier is learned to (approximately) maximize
	the weighted training accuracy at the corresponding iteration, and the weight of each training example is
	updated after every iteration.

	1 General boosting framework 
		A boosting algorithm sequentially learns βt and ht ∈ H for t = 0,..., T−1 and finally constructs 
		a strong classifier as H(x) = sign[∑T−1 t=0 βt ht (x)].

	2 Decision stump 
		A decision stump h ∈ H is a classifier characterized by a triplet (s ∈ {+1, −1}, b ∈ R, 
	d ∈ {0, 1, · · · , D − 1}) such that h(s,b,d)(x) = {
														s, if xd > b,
														−s, otherwise.
													}
		That is, each decision stump function only looks at a single dimension xd of the input vector x, 
	and check whether xd is larger than b. Then s decides which label to predict if xd > b.

	3 AdaBoost AdaBoost 
		A powerful and popular boosting method.

	4 
		Run boosting.sh, which will generate boosting.json.