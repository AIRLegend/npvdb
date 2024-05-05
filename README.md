# npvDB - numpy Vector DB

Inspired by this tweet from the almighty Karpathy I decided to give it a try:

<html>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">np.array<br>people keep reaching for much fancier things way too fast these days</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1647374645316968449?ref_src=twsrc%5Etfw">April 15, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</html>





## TODO Ideas
- [ ] Add tests
- [ ] Add suport for new index systems
- right now there is only one index supported (KMeans). The ability to build new ones based on other techniques such as LSH, Hierarchical, or random projections .

- [ ] Add suport for new search distance metrics (RN only `cosine` is supported)

- [ ] Add support for iterative initialization in case you cannot load `data` in a numpy array. Calling `init` with a generator loading parts of the vector matrix would solve this issue.

- [ ] Dynamic initialization of centroids:
- Right now the user must specify the number of space partitions via `n_worker` parameter. This presents a tradeoff between search speed, memory consumption and query latency. 
- One idea would be to compute several KMeans centroids on each worker and then aggregate them on the "master" process to update the `_search_index`.

- [ ] Add network support for horizontal scaling. Workers could communicate over TCP.


