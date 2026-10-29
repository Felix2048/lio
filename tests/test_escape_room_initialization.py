import os,sys,unittest,itertools,functools
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from lio.alg.er_configs import config_room_tax_n2,config_room_tax_n3
from lio.env.room_symmetric_tax import EscapeRoom as TaxRoom
from lio.env.room_symmetric_baseline import EscapeRoom as BaselineRoom

def room(n,randomize=False,missing=False):
 tf.reset_default_graph();tf.set_random_seed(12340)
 config=(config_room_tax_n2 if n==2 else config_room_tax_n3).get_config()
 if missing:del config.env['randomize']
 else:config.env.randomize=randomize
 return TaxRoom(5,n,config)

class ResetRegression(unittest.TestCase):
 def test_tax_fixed_start_for_both_sizes(self):
  for n in (2,3):
   with self.subTest(n=n):
    env=room(n);np.random.seed(12340)
    for _ in range(40):
     obs=env.reset();self.assertEqual(env.state,[1]*n)
     self.assertEqual(env.steps,0);self.assertFalse(env.solved);self.assertEqual(env.bank,0)
     for x in obs:np.testing.assert_array_equal(x,np.tile([0,1,0],n))
 def test_tax_explicit_random_starts_for_both_sizes(self):
  for n in (2,3):
   with self.subTest(n=n):
    env=room(n,True);np.random.seed(12340)
    states=set()
    for _ in range(80):env.reset();states.add(tuple(env.state))
    self.assertGreater(len(states),1)
    self.assertTrue(all(0<=v<=2 for state in states for v in state))
 def test_tax_evaluation_always_starts_at_start(self):
  for n in (2,3):
   env=room(n,True);np.random.seed(12340)
   for _ in range(20):env.reset(tested=True);self.assertEqual(env.state,[1]*n)
   states=set()
   for _ in range(40):env.reset();states.add(tuple(env.state))
   self.assertGreater(len(states),1)
 def test_missing_tax_flag_defaults_to_fixed(self):
  env=room(3,missing=True)
  for _ in range(20):env.reset();self.assertEqual(env.state,[1,1,1])
 def test_baseline_obeys_initialization_flag(self):
  for n in (2,3):
   for randomize in (False,True):
    with self.subTest(n=n,randomize=randomize):
     env=BaselineRoom(5,n,incentivization_inside_env=False,observe_given=False,randomize=randomize)
     np.random.seed(12340);states=set()
     for _ in range(80):env.reset();states.add(tuple(env.state))
     self.assertEqual(len(states)==1,not randomize)
     if not randomize:self.assertEqual(states,{tuple([1]*n)})
     for _ in range(20):env.reset(tested=True);self.assertEqual(env.state,[1]*n)
 def test_exhaustive_original_reward_upper_bounds(self):
  for n,bound in ((2,9),(3,8)):
   with self.subTest(n=n):
    env=room(n);env.reset();self.assertEqual(env.state,[1]*n)
    actions=list(itertools.product(range(3),repeat=n))
    @functools.lru_cache(None)
    def best(state,remaining):
     if not remaining:return 0.
     scores=[]
     for action in actions:
      env.state=list(state);env.steps=5-remaining;env.solved=False;env.bank=0.
      for actor,pos in zip(env.actors,state):actor.position=pos
      _,_,rewards,done,_=env.step(list(action))
      scores.append(float(np.sum(rewards))+(0 if done else best(tuple(env.state),remaining-1)))
     return max(scores)
    self.assertEqual(best(tuple(env.state),5),bound)

suite=unittest.defaultTestLoader.loadTestsFromTestCase(ResetRegression)
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
