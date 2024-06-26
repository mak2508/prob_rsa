import torch
import pyro
import pyro.distributions as dist


class Prior:
    def __init__(self, lang_file):
        self.lang_specs = eval(open(lang_file).read())
        self.worlds = list(self.lang_specs["worlds"].keys())
        self.n_worlds = len(self.worlds)

        self.contexts = self._make_contexts(self.lang_specs.get("contexts", None), self.n_worlds)
        self.n_contexts = self.contexts.size(0)

        self.utterances = list(self.lang_specs["utterances"].keys())
        self.n_utterances = len(self.utterances)

    def world_prior(self, context=None, obs=None, uniform=False):
        """
        World prior optionally conditioned on context.
        When context is provided, the world prior is normalized amongst the worlds
        that are consistent with the context.
        Args:
            context (optional): torch.tensor - context vector
            uniform (optional): bool - if True, return uniform distribution over worlds
        """
        if uniform or not "world_prior" in self.lang_specs:
            world_probs = torch.ones(self.n_worlds) / self.n_worlds
        else:
            world_probs = torch.tensor(
                [self.lang_specs["world_prior"][w] for w in self.worlds]
            )

        if context is not None:
            world_probs = world_probs * context / torch.sum(world_probs * context)

        if obs != None:
            obs = torch.nonzero(obs).item()

        ix = pyro.sample("world", dist.Categorical(probs=world_probs), obs=obs)
        return torch.nn.functional.one_hot(
            ix, num_classes=self.n_worlds
        ).bool()

    def context_prior(self):
        """
        Context prior
        """
        ix = pyro.sample(
            "context",
            dist.Categorical(probs=torch.ones(self.n_contexts) / self.n_contexts),
        )
        return self.contexts[ix]

    def utterance_prior(self, cost_multiplier):
        """
        Utterance prior that takes into account the cost of utterances
        Args:
            cost_multiplier: float - multiplier for the cost of utterances
        """
        costs = torch.tensor(
            [self.utterance_cost(u) for u in self.utterances], dtype=torch.float
        )

        priors = torch.ones(self.n_utterances) / self.n_utterances
        adjusted_priors = torch.softmax(
            torch.log(priors) - cost_multiplier * costs, dim=0
        )
        ix = pyro.sample("utterance", dist.Categorical(probs=adjusted_priors))
        return self.utterances[ix]

    def qud_prior(self, world, context, obs=None, uniform=True):
        """
        QUD prior conditioned on world and context
        Args:
            world: torch.tensor - world vector
            context: torch.tensor - context vector
            obs (optional): str - observed qud
            uniform (optional): bool - if True, return uniform distribution over quds
        """
        quds = list(self.lang_specs["quds"].keys())
        n_quds = len(quds)
        if obs:
            obs = torch.tensor(quds.index(obs))

        if uniform:
            ix = pyro.sample(
                "qud", dist.Categorical(probs=torch.ones(n_quds) / n_quds), obs=obs
            )
        else:
            # The non uniform case is not used in the original paper.
            # However, the posibility of this is discussed in the original codebase.
            # TODO: Consider how a what it would mean to have a prior on QUDs that is
            # conditioned on the world and context.
            raise NotImplementedError
            
        return quds[ix]

    def utterance_cost(self, utterance):
        return 0.0 if utterance == "_" else 1.0

    def _gen_context_tensor(self, i, n_worlds):
        mask = 2 ** torch.arange(n_worlds)
        return torch.IntTensor([i]).bitwise_and(mask).eq(0).bool()

    def _make_contexts(self, cont_str, n_worlds):
        if cont_str is not None:
            return torch.stack([torch.IntTensor(c).bool() for c in cont_str])
        n_contexts = 2**n_worlds - 1
        return torch.stack(
            [self._gen_context_tensor(i, n_worlds) for i in range(n_contexts)]
        )
