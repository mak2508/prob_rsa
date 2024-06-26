import matplotlib.pyplot as plt

def plot_dist(dist, title, names):
    data = [dist.log_prob(s).exp().item() for s in dist.enumerate_support()]

    ax = plt.subplot(111)
    ax.set_title(title)
    width=0.3
    bins = list(map(lambda x: x-width/2,range(1,len(data)+1)))
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(map(lambda x: x, range(1,len(data)+1))))
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")