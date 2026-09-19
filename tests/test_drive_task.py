import numpy as np
import pytest

from openjev.research.drive_task import (
    DT,
    HORIZON,
    INITIAL_POSE,
    LANDMARKS,
    command,
    make_streams,
    metrics,
    motion,
    private_parameters,
    reference,
    sense,
    transition,
    wrap,
)


def test_motion_and_wrapping():
    np.testing.assert_allclose(motion(np.array([0., 0., 0.]), np.array([1., 0.])), [.1, 0., 0.])
    assert -np.pi <= motion(np.array([0., 0., np.pi-.01]), np.array([0., 1.]))[2] < np.pi
    np.testing.assert_allclose(wrap(np.pi+.2),-np.pi+.2)


def test_reference_derivatives_and_initial_heading():
    for t in (0., 1., 6., 17.):
        p, v, w = reference(t)
        delta = 1e-5
        before = reference(t-delta)
        after = reference(t+delta)
        np.testing.assert_allclose((after[0]-before[0])/(2*delta),v,atol=1e-9)
        angles = [np.arctan2(x[1][1], x[1][0]) for x in (before, after)]
        np.testing.assert_allclose(wrap(angles[1]-angles[0])/(2*delta),w,atol=1e-9)
        assert np.isfinite(p).all()
    np.testing.assert_allclose(command(INITIAL_POSE,0.),[np.sqrt(2)*.66,0.],atol=1e-12)


def test_packets_are_endpoint_measurements_and_missing_values_are_poisoned():
    streams = make_streams(9991)  # Engineering stream, outside scored seeds.
    streams['process_noise'][:]=0
    streams['odometry_noise'][:]=0
    streams['landmark_noise'][:]=0
    streams['initial_params'][:]=0
    streams['outage_phase']=np.array(20)
    pose, odom, packet = transition(INITIAL_POSE, np.array([1.,.2]), streams, 'stationary',0)
    np.testing.assert_allclose(odom,[1.,.2])
    np.testing.assert_allclose(packet['values'][packet['valid'],0],
                               np.linalg.norm(LANDMARKS[packet['valid']]-pose[:2],axis=1))
    missing = sense(pose,np.zeros((6,2)),False)
    assert not missing['valid'].any() and np.isnan(missing['values']).all()


def test_command_integration_is_not_a_hidden_state_oracle():
    streams=make_streams(9992)
    streams['initial_params'][:]=[np.log(.7),np.log(1.2),0.,0.]
    streams['process_noise'][:]=0
    pose, _, _ = transition(INITIAL_POSE,np.array([1.,.2]),streams,'stationary',0)
    assert np.linalg.norm(pose-motion(INITIAL_POSE,np.array([1.,.2]))) > .02


def test_fixed_sensor_calibration_and_independent_gain_switch():
    streams=make_streams(9993)
    switch=int(streams['switch_step'])
    assert 120 <= switch <= 180
    np.testing.assert_array_equal(private_parameters(streams,'gain_switch',switch-1),streams['initial_params'])
    np.testing.assert_array_equal(private_parameters(streams,'gain_switch',switch),streams['changed_params'])
    np.testing.assert_array_equal(streams['initial_params'][2:],streams['changed_params'][2:])
    np.testing.assert_array_equal(private_parameters(streams,'stationary',299),streams['initial_params'])


def test_tracking_score_cannot_be_solved_by_stopping_at_path_intersection():
    stopped=np.tile(INITIAL_POSE,(HORIZON,1))
    result=metrics(stopped,stopped,np.zeros((HORIZON,2)),150,'stationary')
    assert result['tracking_mse'] > 1.
    positions=np.array([np.r_[reference((i+1)*DT)[0],0.] for i in range(HORIZON)])
    assert metrics(positions,positions,np.zeros((HORIZON,2)),150,'stationary')['tracking_mse']==0.


def test_failing_observer_retains_paid_transition_prefix(monkeypatch):
    from openjev.research import drive_observers
    from openjev.research.drive_task import EpisodeFailure, run_episode

    class FailingObserver:
        def __init__(self, *args):
            self.count = 0

        def step(self, action, odometry, packet, oracle_params=None):
            assert oracle_params is None
            assert set(packet) == {'values', 'valid'}
            self.count += 1
            if self.count == 3:
                raise ArithmeticError('engineering injected failure')
            return INITIAL_POSE.copy()

    monkeypatch.setattr(drive_observers, 'Observer', FailingObserver)
    with pytest.raises(EpisodeFailure) as caught:
        run_episode('joint_ekf', 'stationary', make_streams(9994))
    assert caught.value.native_steps == 3
    assert caught.value.trace['commands'].shape == (3, 2)
    assert caught.value.trace['poses'].shape == (3, 3)
    assert caught.value.trace['estimates'].shape == (2, 3)
    assert isinstance(caught.value.__cause__, ArithmeticError)
