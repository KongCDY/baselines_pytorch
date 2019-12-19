import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from ptbaselines.algos.common import torch_utils

try:
    from mpi4py import MPI
    from ptbaselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.model = policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001, eps = 1e-5)   # lr will change later

        self.step = self.model.step
        self.value = self.model.value
        self.initial_state = self.model.initial_state

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.save = self.model.save
        self.load = self.model.load

    def train(self, lr, cliprange, obs, returns, masks, actions, values, old_neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs = obs.float()
        neglogpac, pd = self.model.neglogp(obs, actions)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = torch.mean(pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = self.model.value(obs)
        vpredclip = values + torch.clamp(vpred - values, -cliprange, cliprange)

        # Unclipped value
        vf_losses1 = (vpred - returns)**2
        # Clipped value
        vf_losses2 = (vpredclip - returns)**2

        vf_loss = 0.5*torch.mean(torch.max(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = torch.exp(old_neglogpacs - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

        # Final PG loss
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        approxkl = .5 * torch.mean((neglogpac - old_neglogpacs)**2)
        clipped = (ratio > 1 + cliprange) | (ratio < 1 - cliprange)
        clipfrac = torch.mean(clipped.float())

        # Total loss
        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()
        loss.backward()
        # average_gradients(self.optimizer.param_groups)
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        outputs = torch_utils.toNumpy((pg_loss, vf_loss, entropy, approxkl, clipfrac))
        return outputs
