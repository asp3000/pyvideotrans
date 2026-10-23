# -*- coding: utf-8 -*-
import os
import shutil

from videotrans.configure.config import ROOT_DIR, app_cfg, settings, logger
from videotrans.task.taskcfg import InputFile
import unicodedata
from pathlib import Path
from typing import Tuple, Union

def send_notification(title, message):
    if app_cfg.exec_mode == 'cli':
        print(f'\n*****[{title}]: {message}*****\n')
        return
    # GUI 模式下不再发送系统通知：
    # plyer 的 Windows 通知实现每次调用都会创建一个临时托盘图标窗口，
    # 进程内销毁不彻底，会在系统托盘残留大量 ghost 图标(鼠标悬停后才消失)
    if app_cfg.exit_soft or settings.get('dont_notify', False):
        return
    logger.debug(f'通知已禁用(GUI模式不创建托盘图标): [{title}] {message}')


def format_video(name, target_dir=None)->InputFile:
    from . import help_misc
    raw_pathlib = Path(name)
    raw_basename = raw_pathlib.name
    raw_noextname = raw_pathlib.stem
    ext = raw_pathlib.suffix.lower()[1:]
    raw_dirname = raw_pathlib.parent.resolve().as_posix()

    obj = InputFile(
        name=name,
        dirname=raw_dirname,
        basename=raw_basename,
        noextname=raw_noextname,
        ext=ext
    )

    if target_dir:
        obj.target_dir = Path(f'{target_dir}/{raw_noextname}-{ext}').as_posix()
    _stat = raw_pathlib.stat()
    obj.uuid = help_misc.get_md5(f'{name}-{_stat.st_size}-{_stat.st_mtime}')[:10]
    return obj



