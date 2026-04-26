#!/usr/bin/env python3

"""Tkinter control panel for the CNN regulator."""

from datetime import datetime
import tkinter as tk

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class CNNTargetGUI(Node):
    """Tkinter-based interface for commanding the CNN regulator."""

    BACKGROUND = '#0f172a'
    PANEL = '#111827'
    PANEL_ALT = '#172033'
    BORDER = '#24324a'
    TEXT = '#e5e7eb'
    MUTED = '#94a3b8'
    ACCENT = '#14b8a6'
    ACCENT_DARK = '#0f766e'
    SUCCESS = '#22c55e'
    WARNING = '#f59e0b'
    DANGER = '#ef4444'
    SLIDER_MIN = -3.14
    SLIDER_MAX = 3.14
    FONT = 'DejaVu Sans'
    MONO = 'DejaVu Sans Mono'

    def __init__(self):
        super().__init__('cnn_target_gui')

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        self.targets = {
            'home': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            'training': np.array([1.0, -1.0, 1.0, -1.0, 1.0, 0.5], dtype=float),
            'extended': np.array([2.0, -0.5, 0.5, -1.0, 1.0, 0.5], dtype=float),
            'compact': np.array([0.5, -1.5, 1.5, -1.0, 0.5, 0.2], dtype=float),
            'demo': np.array([1.5, -0.8, 0.8, -1.2, 1.2, 0.3], dtype=float),
        }

        self.current_position = np.zeros(6, dtype=float)
        self.current_velocity = np.zeros(6, dtype=float)
        self.target_position = self.targets['training'].copy()
        self.has_joint_state = False
        self.last_joint_state_stamp = '--'
        self._closing = False
        self._suspend_slider_events = False
        self._spin_error_logged = False

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.target_pub = self.create_publisher(
            Float64MultiArray,
            '/cnn_regulator/target_position',
            10,
        )

        self.root = tk.Tk()
        self.root.title('CNN Regulator Control Panel')
        self.root.geometry('1300x800')
        self.root.minsize(1160, 720)
        self.root.configure(bg=self.BACKGROUND)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.auto_publish_var = tk.BooleanVar(master=self.root, value=False)
        self.connection_var = tk.StringVar(master=self.root, value='Waiting for /joint_states')
        self.status_var = tk.StringVar(master=self.root, value='Idle')
        self.error_norm_var = tk.StringVar(master=self.root, value='--')
        self.timestamp_var = tk.StringVar(master=self.root, value='--')
        self.target_vector_var = tk.StringVar(master=self.root, value=self._format_vector(self.target_position))
        self.command_var = tk.StringVar(master=self.root, value='Command: ready')

        self.current_value_vars = [tk.StringVar(master=self.root, value='--') for _ in self.joint_names]
        self.velocity_value_vars = [tk.StringVar(master=self.root, value='--') for _ in self.joint_names]
        self.error_value_vars = [tk.StringVar(master=self.root, value='--') for _ in self.joint_names]
        self.target_value_vars = [tk.StringVar(master=self.root, value=f'{value:+0.3f}') for value in self.target_position]
        self.slider_vars = [tk.DoubleVar(master=self.root, value=float(value)) for value in self.target_position]
        self.slider_widgets = []

        self._build_ui()
        self._apply_target_to_sliders(self.target_position)
        self._refresh_display()
        self.root.after(50, self._poll_ros)

    def _build_ui(self):
        top_bar = tk.Frame(self.root, bg=self.BACKGROUND, padx=22, pady=18)
        top_bar.pack(fill='x')

        tk.Label(
            top_bar,
            text='CNN REGULATOR CONTROL PANEL',
            bg=self.BACKGROUND,
            fg=self.TEXT,
            font=(self.FONT, 22, 'bold'),
            anchor='w',
        ).pack(fill='x')
        tk.Label(
            top_bar,
            text='Set a 6-joint target, then let the CNN regulator move the arm in Gazebo.',
            bg=self.BACKGROUND,
            fg=self.MUTED,
            font=(self.FONT, 10),
            anchor='w',
        ).pack(fill='x', pady=(4, 0))

        badge_row = tk.Frame(top_bar, bg=self.BACKGROUND)
        badge_row.pack(fill='x', pady=(12, 0))

        self.connection_badge = self._create_badge(badge_row, self.connection_var, self.WARNING)
        self.connection_badge.pack(side='left')
        self.status_badge = self._create_badge(badge_row, self.status_var, self.ACCENT)
        self.status_badge.pack(side='left', padx=(10, 0))
        self.error_badge = self._create_badge(badge_row, self.error_norm_var, self.ACCENT_DARK)
        self.error_badge.pack(side='left', padx=(10, 0))
        self.timestamp_badge = self._create_badge(badge_row, self.timestamp_var, self.PANEL_ALT)
        self.timestamp_badge.pack(side='right')

        body = tk.Frame(self.root, bg=self.BACKGROUND, padx=22, pady=10)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        state_card = self._create_card(body, 'Live State', 'Read-only telemetry from /joint_states')
        state_card.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        control_card = self._create_card(body, 'Target Control', 'Drag the sliders, then publish the target to the CNN regulator')
        control_card.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

        self._build_state_panel(state_card)
        self._build_control_panel(control_card)

        footer = tk.Frame(self.root, bg=self.BACKGROUND, padx=22, pady=14)
        footer.pack(fill='x')
        tk.Label(
            footer,
            textvariable=self.target_vector_var,
            bg=self.BACKGROUND,
            fg=self.TEXT,
            font=(self.MONO, 11),
            anchor='w',
        ).pack(fill='x')
        tk.Label(
            footer,
            textvariable=self.command_var,
            bg=self.BACKGROUND,
            fg=self.ACCENT,
            font=(self.MONO, 10, 'bold'),
            anchor='w',
        ).pack(fill='x', pady=(4, 0))

    def _create_card(self, parent, title, subtitle):
        card = tk.Frame(
            parent,
            bg=self.PANEL,
            highlightbackground=self.BORDER,
            highlightthickness=1,
            bd=0,
        )
        header = tk.Frame(card, bg=self.PANEL)
        header.pack(fill='x', padx=16, pady=(14, 10))

        tk.Label(
            header,
            text=title,
            bg=self.PANEL,
            fg=self.TEXT,
            font=(self.FONT, 13, 'bold'),
            anchor='w',
        ).pack(fill='x')
        tk.Label(
            header,
            text=subtitle,
            bg=self.PANEL,
            fg=self.MUTED,
            font=(self.FONT, 9),
            anchor='w',
        ).pack(fill='x', pady=(3, 0))
        return card

    def _create_badge(self, parent, text_var, fill_color):
        return tk.Label(
            parent,
            textvariable=text_var,
            bg=fill_color,
            fg='white',
            font=(self.FONT, 10, 'bold'),
            padx=12,
            pady=5,
            relief='flat',
        )

    def _build_state_panel(self, parent):
        table_frame = tk.Frame(parent, bg=self.PANEL)
        table_frame.pack(fill='both', expand=True, padx=14, pady=(0, 12))

        header_bg = self.PANEL_ALT
        header_row = tk.Frame(table_frame, bg=header_bg)
        header_row.pack(fill='x')

        headers = [
            ('Joint', 28),
            ('Current', 12),
            ('Velocity', 12),
            ('Target', 12),
            ('Error', 12),
        ]
        for column_index, (label_text, width) in enumerate(headers):
            tk.Label(
                header_row,
                text=label_text,
                bg=header_bg,
                fg=self.TEXT,
                font=(self.MONO, 10, 'bold'),
                width=width,
                anchor='w',
                padx=8,
                pady=6,
            ).grid(row=0, column=column_index, sticky='w')

        for index, joint_name in enumerate(self.joint_names):
            row_bg = self.PANEL if index % 2 == 0 else self.PANEL_ALT
            row = tk.Frame(table_frame, bg=row_bg)
            row.pack(fill='x')

            tk.Label(
                row,
                text=joint_name,
                bg=row_bg,
                fg=self.TEXT,
                font=(self.FONT, 10, 'bold'),
                width=28,
                anchor='w',
                padx=8,
                pady=7,
            ).grid(row=0, column=0, sticky='w')

            tk.Label(
                row,
                textvariable=self.current_value_vars[index],
                bg=row_bg,
                fg=self.TEXT,
                font=(self.MONO, 10),
                width=12,
                anchor='e',
                padx=8,
            ).grid(row=0, column=1, sticky='e')

            tk.Label(
                row,
                textvariable=self.velocity_value_vars[index],
                bg=row_bg,
                fg=self.TEXT,
                font=(self.MONO, 10),
                width=12,
                anchor='e',
                padx=8,
            ).grid(row=0, column=2, sticky='e')

            tk.Label(
                row,
                textvariable=self.target_value_vars[index],
                bg=row_bg,
                fg=self.ACCENT,
                font=(self.MONO, 10, 'bold'),
                width=12,
                anchor='e',
                padx=8,
            ).grid(row=0, column=3, sticky='e')

            tk.Label(
                row,
                textvariable=self.error_value_vars[index],
                bg=row_bg,
                fg=self.SUCCESS,
                font=(self.MONO, 10, 'bold'),
                width=12,
                anchor='e',
                padx=8,
            ).grid(row=0, column=4, sticky='e')

        summary = tk.Frame(parent, bg=self.PANEL_ALT, padx=12, pady=12)
        summary.pack(fill='x', padx=14, pady=(0, 14))
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        summary.columnconfigure(2, weight=1)

        self.summary_error_label = self._create_summary_field(summary, 'Error norm', self.error_norm_var, 0)
        self.summary_time_label = self._create_summary_field(summary, 'Last update', self.timestamp_var, 1)
        self.summary_connection_label = self._create_summary_field(summary, 'Connection', self.connection_var, 2)

    def _create_summary_field(self, parent, title, value_var, column_index):
        wrapper = tk.Frame(parent, bg=self.PANEL_ALT)
        wrapper.grid(row=0, column=column_index, sticky='ew', padx=4)

        tk.Label(
            wrapper,
            text=title,
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            font=(self.FONT, 9),
            anchor='w',
        ).pack(fill='x')
        label = tk.Label(
            wrapper,
            textvariable=value_var,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            font=(self.MONO, 11, 'bold'),
            anchor='w',
        )
        label.pack(fill='x', pady=(2, 0))
        return label

    def _build_control_panel(self, parent):
        sliders_frame = tk.Frame(parent, bg=self.PANEL)
        sliders_frame.pack(fill='both', expand=True, padx=14, pady=(0, 10))

        for index, joint_name in enumerate(self.joint_names):
            row_bg = self.PANEL if index % 2 == 0 else self.PANEL_ALT
            row = tk.Frame(sliders_frame, bg=row_bg, padx=12, pady=10)
            row.pack(fill='x', pady=4)

            title_row = tk.Frame(row, bg=row_bg)
            title_row.pack(fill='x')

            tk.Label(
                title_row,
                text=joint_name,
                bg=row_bg,
                fg=self.TEXT,
                font=(self.FONT, 10, 'bold'),
                anchor='w',
            ).pack(side='left')
            tk.Label(
                title_row,
                textvariable=self.target_value_vars[index],
                bg=row_bg,
                fg=self.ACCENT,
                font=(self.MONO, 10, 'bold'),
                width=10,
                anchor='e',
            ).pack(side='right')

            scale = tk.Scale(
                row,
                from_=self.SLIDER_MIN,
                to=self.SLIDER_MAX,
                orient='horizontal',
                resolution=0.01,
                showvalue=0,
                variable=self.slider_vars[index],
                command=lambda value, joint_index=index: self._on_slider_change(joint_index, value),
                bg=row_bg,
                fg=self.TEXT,
                troughcolor='#334155',
                activebackground=self.ACCENT,
                highlightthickness=0,
                bd=0,
                length=420,
            )
            scale.pack(fill='x', pady=(8, 0))
            self.slider_widgets.append(scale)

        options_frame = tk.Frame(parent, bg=self.PANEL)
        options_frame.pack(fill='x', padx=14, pady=(0, 14))

        self.auto_publish_check = tk.Checkbutton(
            options_frame,
            text='Auto publish while dragging',
            variable=self.auto_publish_var,
            bg=self.PANEL,
            fg=self.TEXT,
            selectcolor=self.PANEL_ALT,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            font=(self.FONT, 10),
            bd=0,
            highlightthickness=0,
        )
        self.auto_publish_check.pack(anchor='w')

        button_row_1 = tk.Frame(options_frame, bg=self.PANEL)
        button_row_1.pack(fill='x', pady=(10, 0))

        self._create_action_button(button_row_1, 'Send Target', self.send_target, self.ACCENT, 'white').pack(side='left', expand=True, fill='x', padx=(0, 6))
        self._create_action_button(button_row_1, 'Hold Current', self.hold_current_target, self.SUCCESS, 'white').pack(side='left', expand=True, fill='x', padx=6)
        self._create_action_button(button_row_1, 'Copy Current', self.copy_current_to_sliders, self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=(6, 0))

        button_row_2 = tk.Frame(options_frame, bg=self.PANEL)
        button_row_2.pack(fill='x', pady=(10, 0))

        self._create_action_button(button_row_2, 'Home', lambda: self.load_preset('home'), self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=(0, 6))
        self._create_action_button(button_row_2, 'Training', lambda: self.load_preset('training'), self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=6)
        self._create_action_button(button_row_2, 'Extended', lambda: self.load_preset('extended'), self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=(6, 0))

        button_row_3 = tk.Frame(options_frame, bg=self.PANEL)
        button_row_3.pack(fill='x', pady=(10, 0))

        self._create_action_button(button_row_3, 'Compact', lambda: self.load_preset('compact'), self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=(0, 6))
        self._create_action_button(button_row_3, 'Demo', lambda: self.load_preset('demo'), self.PANEL_ALT, self.TEXT).pack(side='left', expand=True, fill='x', padx=6)
        self._create_action_button(button_row_3, 'Exit', self.on_close, self.DANGER, 'white').pack(side='left', expand=True, fill='x', padx=(6, 0))

    def _create_action_button(self, parent, text, command, background, foreground):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=background,
            activeforeground=foreground,
            bd=0,
            relief='flat',
            font=(self.FONT, 10, 'bold'),
            padx=10,
            pady=8,
            cursor='hand2',
        )

    def _on_slider_change(self, joint_index, value):
        if self._suspend_slider_events:
            return

        numeric_value = float(value)
        self.target_position[joint_index] = numeric_value
        self.target_value_vars[joint_index].set(f'{numeric_value:+0.3f}')
        self.target_vector_var.set(self._format_vector(self.target_position))

        if self.auto_publish_var.get():
            self.publish_target(self.target_position)
            self.command_var.set(f'Command: auto-published target {self._format_vector(self.target_position)}')

    def _apply_target_to_sliders(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (6,):
            return

        self._suspend_slider_events = True
        try:
            for index, value in enumerate(target):
                self.slider_widgets[index].set(float(value))
                self.target_value_vars[index].set(f'{float(value):+0.3f}')
        finally:
            self._suspend_slider_events = False

        self.target_position = target.copy()
        self.target_vector_var.set(self._format_vector(self.target_position))

    def _format_vector(self, values):
        values = np.asarray(values, dtype=float)
        return '[' + ', '.join(f'{value:+0.3f}' for value in values) + ']'

    def _set_command(self, message):
        self.command_var.set(f'Command: {message}')

    def publish_target(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (6,):
            self._set_command('invalid target shape')
            return False

        command_msg = Float64MultiArray()
        command_msg.data = target.tolist()
        self.target_pub.publish(command_msg)
        return True

    def send_target(self):
        self._apply_target_to_sliders(self.target_position)
        if self.publish_target(self.target_position):
            self._set_command(f'sent target {self._format_vector(self.target_position)}')

    def load_preset(self, preset_name):
        target = self.targets.get(preset_name)
        if target is None:
            self._set_command(f'unknown preset: {preset_name}')
            return

        self._apply_target_to_sliders(target)
        if self.publish_target(target):
            self._set_command(f'sent preset {preset_name.upper()} {self._format_vector(target)}')

    def copy_current_to_sliders(self):
        if not self.has_joint_state:
            self._set_command('waiting for current joint state')
            return

        self._apply_target_to_sliders(self.current_position)
        self._set_command('copied current joint state to sliders')

    def hold_current_target(self):
        if not self.has_joint_state:
            self._set_command('waiting for current joint state')
            return

        self._apply_target_to_sliders(self.current_position)
        if self.publish_target(self.current_position):
            self._set_command(f'holding current pose {self._format_vector(self.current_position)}')

    def joint_state_callback(self, msg):
        try:
            indices = [msg.name.index(name) for name in self.joint_names]
            positions = np.array([msg.position[idx] for idx in indices], dtype=float)
        except ValueError:
            return

        velocities = np.zeros(6, dtype=float)
        if len(msg.velocity) >= max(indices) + 1:
            try:
                velocities = np.array([msg.velocity[idx] for idx in indices], dtype=float)
            except (IndexError, TypeError):
                velocities = np.zeros(6, dtype=float)

        self.current_position = positions
        self.current_velocity = velocities
        self.has_joint_state = True
        self.last_joint_state_stamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    def _poll_ros(self):
        if self._closing or not rclpy.ok():
            return

        try:
            rclpy.spin_once(self, timeout_sec=0.0)
        except Exception as exc:  # pragma: no cover - defensive for GUI loop
            if not self._spin_error_logged:
                self.get_logger().warn(f'ROS poll failed: {exc}')
                self._spin_error_logged = True

        self._refresh_display()
        self.root.after(50, self._poll_ros)

    def _refresh_display(self):
        if self.has_joint_state:
            error = self.target_position - self.current_position
            error_norm = float(np.linalg.norm(error))
            max_velocity = float(np.max(np.abs(self.current_velocity)))

            for index in range(len(self.joint_names)):
                self.current_value_vars[index].set(f'{self.current_position[index]:+0.3f}')
                self.velocity_value_vars[index].set(f'{self.current_velocity[index]:+0.3f}')
                self.error_value_vars[index].set(f'{error[index]:+0.3f}')
                self.target_value_vars[index].set(f'{self.target_position[index]:+0.3f}')

            if error_norm < 0.1:
                status_text = 'At target'
                status_color = self.SUCCESS
            elif max_velocity > 0.01:
                status_text = 'Moving'
                status_color = self.ACCENT
            else:
                status_text = 'Idle'
                status_color = self.WARNING

            self.connection_var.set('ROS: connected')
            self.status_var.set(status_text)
            self.error_norm_var.set(f'Error norm: {error_norm:0.4f} rad')
            self.timestamp_var.set(f'Last update: {self.last_joint_state_stamp}')

            self.connection_badge.configure(bg=self.SUCCESS)
            self.status_badge.configure(bg=status_color)
            self.error_badge.configure(bg=self.ACCENT_DARK)
            self.timestamp_badge.configure(bg=self.PANEL_ALT)
        else:
            self.connection_var.set('Waiting for /joint_states')
            self.status_var.set('Idle')
            self.error_norm_var.set('Error norm: --')
            self.timestamp_var.set('Last update: --')

            self.connection_badge.configure(bg=self.WARNING)
            self.status_badge.configure(bg=self.ACCENT)
            self.error_badge.configure(bg=self.ACCENT_DARK)
            self.timestamp_badge.configure(bg=self.PANEL_ALT)

    def on_close(self):
        if self._closing:
            return

        self._closing = True
        try:
            self.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)

    try:
        gui = CNNTargetGUI()
        gui.run()
    except tk.TclError as exc:
        print(f'Unable to start the GUI: {exc}')
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()