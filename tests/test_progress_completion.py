import unittest
from unittest.mock import Mock, patch

from utils.progress_manager import ProgressManager


class ProgressCompletionTests(unittest.TestCase):
    def test_scheduled_completion_updates_real_panel_fields(self):
        # No Tk window; only widgets/scheduler are replaced, methods are real.
        manager = ProgressManager.__new__(ProgressManager)
        manager.parent = Mock()
        pending = []
        manager.parent.after_safe.side_effect = lambda delay, callback: pending.append(callback)
        manager._is_active = True
        manager._stages = ['Prepare', 'Wavelet', 'Extrema', 'Save']
        manager._started_at = 100.0
        manager._timer_callback_id = 'timer-1'
        for name in ('state_badge', 'progress_label', 'stage_label', 'timer_label',
                     'overall_progress_bar', 'stage_progress_bar',
                     'overall_percent_label', 'stage_percent_label', 'completion_actions'):
            setattr(manager, name, Mock())
        actions = [Mock(), Mock(), Mock()]
        manager.finish_run('Завершено: 1 задач', result_callback=actions[0],
                           folder_callback=actions[1], history_callback=actions[2])
        self.assertEqual(len(pending), 1)
        manager.state_badge.configure.assert_not_called()
        with patch('utils.progress_manager.time.monotonic', return_value=165.0):
            pending.pop()()
        self.assertEqual(manager.state_badge.configure.call_args.kwargs['text'], '● ГОТОВО')
        self.assertEqual(manager.progress_label.configure.call_args.kwargs['text'], 'Завершено: 1 задач')
        manager.stage_label.configure.assert_called_once_with(text='Все этапы завершены · 4 из 4')
        manager.overall_progress_bar.set.assert_called_once_with(1.0)
        manager.stage_progress_bar.set.assert_called_once_with(1.0)
        manager.overall_percent_label.configure.assert_called_once_with(text='100%')
        manager.stage_percent_label.configure.assert_called_once_with(text='100%')
        manager.timer_label.configure.assert_called_once_with(text='01:05')
        manager.parent.after_cancel_safe.assert_called_once_with('timer-1')
        self.assertIsNone(manager._timer_callback_id)
        self.assertIsNone(manager._started_at)
        self.assertTrue(manager._is_active)  # Journal remains active.
        manager.completion_actions.pack.assert_called_once()
        for callback in (manager._result_callback, manager._folder_callback, manager._history_callback):
            manager._call_action(callback)
        for action in actions:
            action.assert_called_once_with()
        manager._tick_timer()  # A stale queued tick must not restart the timer.
        self.assertEqual(pending, [])


if __name__ == '__main__':
    unittest.main()
