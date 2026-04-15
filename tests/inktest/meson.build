inktests_riscv64 = [
  'test-pwm-semantics',
]

inktest_executables = {}

foreach dir : target_dirs
  if not dir.endswith('-softmmu')
    continue
  endif

  target_base = dir.split('-')[0]
  target_inktests = get_variable('inktests_' + target_base, [])

  if target_inktests.length() == 0
    continue
  endif

  inktest_emulator = emulators['qemu-system-' + target_base]

  test_deps = roms
  inktest_env = environment()
  if have_tools
    inktest_env.set('QTEST_QEMU_IMG', './qemu-img')
    test_deps += [qemu_img]
  endif
  inktest_env.set('G_TEST_DBUS_DAEMON',
                  meson.project_source_root() / 'tests/dbus-vmstate-daemon.sh')
  inktest_env.set('QTEST_QEMU_BINARY', './qemu-system-' + target_base)
  if have_tools and have_vhost_user_blk_server
    inktest_env.set('QTEST_QEMU_STORAGE_DAEMON_BINARY',
                    './storage-daemon/qemu-storage-daemon')
    test_deps += [qsd]
  endif

  inktest_env.set('PYTHON', python.full_path())

  foreach test_name : target_inktests
    if not inktest_executables.has_key(test_name)
      inktest_executables += {
        test_name: executable(test_name,
                              [test_name + '.c'],
                              dependencies: [qemuutil, qos])
      }
    endif

    test('inktest-@0@/@1@'.format(target_base, test_name),
         inktest_executables[test_name],
         depends: [test_deps, inktest_emulator, emulator_modules],
         env: inktest_env,
         args: ['--tap', '-k'],
         protocol: 'tap',
         timeout: 60,
         priority: 60,
         suite: ['inktest', 'inktest-' + target_base])
  endforeach
endforeach
