import torch
import pyro
import pyro.poutine as poutine

from priors import Prior
from search_inference import HashingMarginal, Search, memoize


def Marginal(fn):
    return memoize(lambda *args, **kwargs: HashingMarginal(Search(fn).run(*args, **kwargs)))

class RSA:
    def __init__(self, lang_file):
        self.priors = Prior(lang_file)
        self.lang_specs = self.priors.lang_specs

        self.utterance_dict = self._gen_utterance_dict()

        self.qud_fns = self._gen_qud_fn_dict()

    @Marginal
    def literal_listener(self, utterance, context):
        """
        Args:
            utterance: str - utterance
            context: torch.tensor - context vector
        """
        world = self.priors.world_prior()
        possible_worlds = self.meaning(utterance)

        pyro.factor(
            "literal_meaning",
            0.0 if torch.any(world & context & possible_worlds) else -999999.0,
        )
        return world

    @Marginal
    def speaker(self, world, qud, context, alpha=10e2, cost_multiplier=0.02):
        """
        Args:
            world: torch.tensor - world state
            qud: str - question under discussion
            context: torch.tensor - context vector
            alpha: float - scale factor for softmax
            cost_multiplier: float - multiplier for the cost of utterances
        """
        qudValue = self.qud_fns[qud](world)

        with poutine.scale(scale=torch.tensor(alpha)):
            utterance = self.priors.utterance_prior(cost_multiplier)
            literal_marginal = self.literal_listener(utterance, context)
            projected_literal = self.project(literal_marginal, qud)
            pyro.sample("literal_listener", projected_literal, obs=qudValue)
        return utterance

    @Marginal
    def pragmatic_listener(self, utterance, qud, output_type="world"):
        """
        Args:
            utterance: str - utterance
            qud: str - question under discussion
            output_type: str - "world" or "context"
        """
        context = self.priors.context_prior()
        world = self.priors.world_prior(context, uniform=True)

        self.priors.qud_prior(world, context, obs=qud, uniform=True)

        pyro.sample("speaker", self.speaker(world, qud, context), obs=utterance)

        if output_type == "world":
            return world
        elif output_type == "context":
            return context
        else:
            raise ValueError("Invalid output_type")

    def meaning(self, utterance):
        """
        Gives the meaning (possible worlds) of an utterance
        Args:
            utterance: str - utterance
        """
        if utterance not in self.utterance_dict.keys():
            return self.utterance_dict["_"]
        return self.utterance_dict[utterance]
    
    @Marginal
    def project(self, dist, qud):
        """
        Projects world distribution value onto QUD alternatives
        Args:
            dist: pyro.distributions - distribution to project
            qud: str - question under discussion
        """
        v = pyro.sample("proj", dist)
        return self.qud_fns[qud](v)

    def get_world_labels(self):
        return self.lang_specs["worlds"].values()

    def get_utterance_labels(self):
        return self.lang_specs["utterances"].keys()

    def get_qud_labels(self):
        return self.lang_specs["quds"].keys()

    def get_alternatives_labels(self, qud):
        return self.lang_specs["quds"][qud].keys()
    
    def get_context_labels(self):
        return [row.tolist() for row in self.priors.contexts]

    def _gen_utterance_dict(self):
        """
        Generate dictionary mapping of utterance to interpreted possible worlds
        """
        utterance_dict = {
            key: torch.tensor(value)
            for key, value in self.lang_specs["utterances"].items()
        }
        return utterance_dict

    def _gen_qud_fn_dict(self):
        """
        Generate dictionary mapping of QUD to QUD functions,
        which return the alternative that a world satisfies
        given the QUD
        """
        def get_alternative(state, alternatives):
            for i, alternative in enumerate(alternatives):
                if torch.any(state & torch.tensor(alternative)):
                    return torch.nn.functional.one_hot(
                        torch.tensor(i), num_classes=len(alternatives)
                    ).bool()

        quds = self.lang_specs["quds"]

        qud_fn_dict = {}
        for qud in quds:
            alternatives = quds[qud].values()
            qud_fn_dict[qud] = lambda state: get_alternative(state, alternatives)

        return qud_fn_dict


if __name__ == "__main__":
    fgs = RSA("olympic_sprinter_simple.json")
    utterance = "Olympic sprinter"
    qud = "profession"
    context = torch.tensor([True, True, True, True])
    world = torch.tensor([False, False, True, False])
    print(fgs.pragmatic_listener(utterance, qud, output_type="world"))
    print("tst")