const { app, BrowserWindow, ipcMain, dialog } = require('electron/main')
const { spawn } = require('child_process');
const path = require('node:path')
const fs = require('node:fs')
const os = require('node:os')

let mainWindow = null
let aiWindow = null
try {
    require('electron-reloader')(module)
  } catch (_) {}
function createWindow () {
  const win = new BrowserWindow({
    width: 600,
    height: 500,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    },
    // frame:false
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: 'rgba(0,0,0,0)',
      height: 35,
      symbolColor: 'white'
    }
  })

  win.loadFile('shouye.html')
  mainWindow = win

  // 进入功能模块时恢复窗口尺寸并居中
  win.webContents.on('did-finish-load', () => {
    const url = win.webContents.getURL();
    if (url.includes('dading.html') || url.includes('dicom.html')) {
      win.setSize(1264, 812);
      win.center();
    } else if (url.includes('shouye.html')) {
      win.setSize(600, 500);
      win.center();
    }
  });

  // 按F12打开/关闭开发者工具
  win.webContents.on('before-input-event', (event, input) => {
    if (input.key === 'F12') {
      win.webContents.toggleDevTools()
      event.preventDefault()
    }
  })
}
async function handleresult(args, modelPath){
  return new Promise(async (resolve,reject)=>{
    const fpath = args;
    const jsonFpath = JSON.stringify(fpath);
    // 按患者 ID 建子文件夹保存（如 savestl/0012/）
    const patientId = modelPath ? path.basename(path.dirname(modelPath)) : 'default';
    const saveDir = path.join(__dirname, 'savestl', patientId);
    if (!fs.existsSync(saveDir)) fs.mkdirSync(saveDir, { recursive: true });

    console.log('[handleresult] args:', args, 'modelPath:', modelPath, 'jsonFpath:', jsonFpath, 'saveDir:', saveDir);
    try{
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', ['./MasterWuVtkStlMaker.py', jsonFpath, modelPath, saveDir], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      let stdoutBuf = Buffer.alloc(0)
      let stderrBuf = Buffer.alloc(0)
      let lastProgressLine = ''
      let stdoutResidual = ''
      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        // 用残余缓冲区处理跨 chunk 的不完整行，避免重复解析旧行
        const text = stdoutResidual + data.toString();
        const lines = text.split('\n');
        stdoutResidual = lines.pop(); // 最后一个元素可能不完整，留待下次
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const obj = JSON.parse(trimmed);
            if (obj.progress !== undefined) {
              lastProgressLine = trimmed;
              console.log(`[生成进度] ${obj.progress}% - ${obj.message}`);
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('generate-progress', obj);
              }
            }
          } catch (e) { /* 非 JSON 行，跳过 */ }
        }
      });
      pythonProcess.stderr.on('data', (data) => {
        stderrBuf = Buffer.concat([stderrBuf, data]);
        console.log('[Python stderr]', data.toString().trim());
      });
      pythonProcess.on('close', (code) => {
        console.log('[handleresult] Python退出 code:', code);
        console.log('[handleresult] stdout:', stdoutBuf.toString().trim());
        console.log('[handleresult] stderr:', stderrBuf.toString().trim().slice(-1000));
        if (code !== 0) {
          reject(new Error(`Python进程异常退出 code=${code}: ${stderrBuf.toString().trim().slice(-300)}`));
          return;
        }
        // 返回固定路径 + Python 原始输出（过滤掉 JSON 进度行，只保留最后3个参数行）
        const text = stdoutBuf.toString().trim();
        const paramLines = text.split('\n').filter(line => {
          // 只过滤 JSON 对象行（含 progress 字段的进度行），保留数字参数行
          try { const obj = JSON.parse(line); return !(obj && typeof obj === 'object' && obj.progress !== undefined); } catch (e) { return true; }
        }).slice(-3);  // 只取最后3行（length, angle1, angle2）
        const result = paramLines.length > 0 ? paramLines.join('\n') + '\n' + saveDir : saveDir;
        // 从 mesh.cfg 读取原始旋转参数，附加到结果中
        let rawAngle1, rawAngle2;
        try {
          const cfgPath2 = path.join(saveDir, 'mesh.cfg');
          if (fs.existsSync(cfgPath2)) {
            const cfgText = fs.readFileSync(cfgPath2, 'utf-8');
            const m1 = cfgText.match(/raw_angle1\s*=\s*([-\d.eE]+)/);
            const m2 = cfgText.match(/raw_angle2\s*=\s*([-\d.eE]+)/);
            if (m1) rawAngle1 = parseFloat(m1[1]);
            if (m2) rawAngle2 = parseFloat(m2[1]);
            console.log('[handleresult] 原始旋转参数:', rawAngle1, rawAngle2);
          }
        } catch(e) {}
        resolve({ result, raw_angle1: rawAngle1, raw_angle2: rawAngle2 });
      });
    } catch (error) {
      reject(error);
    }
  })
}
app.whenReady().then(() => {
  // 打开文件对话框：浏览文件夹内dcm文件，选中任意一个即加载该文件夹所有dcm
  ipcMain.handle('open-dicom-folder', async () => {
    const win = BrowserWindow.getFocusedWindow();
    const result = await dialog.showOpenDialog(win, {
      properties: ['openFile', 'multiSelections'],
      title: '选择DICOM文件',
      buttonLabel: '打开',
      filters: [
        { name: 'DICOM文件', extensions: ['dcm', 'dicom'] },
        { name: '所有文件', extensions: ['*'] }
      ]
    });
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true, files: [] };
    }
    // 取选中文件所在目录，自动加载该目录下所有dcm文件
    const dirPath = path.dirname(result.filePaths[0]);
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    const dcmFiles = [];
    for (const entry of entries) {
      if (entry.isFile() && (entry.name.toLowerCase().endsWith('.dcm') || entry.name.toLowerCase().endsWith('.dicom'))) {
        const filePath = path.join(dirPath, entry.name);
        const data = fs.readFileSync(filePath);
        dcmFiles.push({ name: entry.name, data: new Uint8Array(data).buffer });
      }
    }
    return { canceled: false, files: dcmFiles, dirPath: dirPath };
  });

  // 打开STL文件对话框并直接读取文件内容
  ipcMain.handle('open-stl-file', async () => {
    const win = BrowserWindow.getFocusedWindow();
    const result = await dialog.showOpenDialog(win, {
      properties: ['openFile'],
      title: '选择STL文件',
      buttonLabel: '打开',
      filters: [
        { name: 'STL文件', extensions: ['stl'] },
        { name: '所有文件', extensions: ['*'] }
      ]
    });
    if (!result.canceled && result.filePaths.length > 0) {
      try {
        const data = fs.readFileSync(result.filePaths[0]);
        result.fileData = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      } catch (error) {
        console.error('读取STL文件失败:', error);
      }
    }
    return result;
  });

  // 打开NIfTI文件：选文件 → Python直接读取nii.gz → 写入raw文件 → 返回元数据
  ipcMain.handle('open-nifti-file', async (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    const result = await dialog.showOpenDialog(win, {
      properties: ['openFile'],
      title: '选择NIfTI文件',
      buttonLabel: '打开',
      filters: [
        { name: 'NIfTI文件', extensions: ['nii', 'gz'] },
        { name: '所有文件', extensions: ['*'] }
      ]
    });
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true };
    }
    const niftiPath = result.filePaths[0];
    const rawDir = path.join(path.dirname(niftiPath), '_nifti_raw');

    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './nifti_reader.py', niftiPath, rawDir
      ]);
      let stdoutBuf = Buffer.alloc(0);
      let finalResult = null;

      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        const text = stdoutBuf.toString();
        const lines = text.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line.trim());
              if (progress.progress >= 100) {
                try { finalResult = JSON.parse(progress.message); } catch (e) {}
              }
              if (win && !win.isDestroyed()) {
                win.webContents.send('conversion-progress', progress);
              }
            } catch (e) {}
          }
        }
        stdoutBuf = Buffer.alloc(0);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.log('[nifti_reader stderr]', data.toString().trim());
      });

      pythonProcess.on('close', (code) => {
        if (code === 0 && finalResult) {
          resolve({
            canceled: false,
            rawDir: rawDir,
            niftiDir: path.dirname(niftiPath),
            niftiPath: niftiPath,
            shape: finalResult.shape,
            spacing: finalResult.spacing
          });
        } else {
          reject(new Error('NIfTI读取失败'));
        }
      });
    });
  });

  // 直接读取NIfTI文件（不弹对话框，用于加载分割结果）
  ipcMain.handle('read-nifti-direct', async (event, niftiPath) => {
    const rawDir = path.join(path.dirname(niftiPath), '_seg_raw');
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './nifti_reader.py', niftiPath, rawDir, '--raw'
      ], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      let stdoutBuf = Buffer.alloc(0);
      let finalResult = null;
      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        const text = stdoutBuf.toString();
        const lines = text.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line.trim());
              if (progress.progress >= 100) {
                try { finalResult = JSON.parse(progress.message); } catch (e) {}
              }
            } catch (e) {}
          }
        }
        stdoutBuf = Buffer.alloc(0);
      });
      pythonProcess.stderr.on('data', (data) => {
        console.log('[read-nifti-direct stderr]', data.toString().trim());
      });
      pythonProcess.on('close', (code) => {
        if (code === 0 && finalResult) {
          const metaPath2 = path.join(rawDir, 'meta.json');
          const rawPath = path.join(rawDir, 'volume.raw');
          const meta = JSON.parse(fs.readFileSync(metaPath2, 'utf-8'));
          const rawData = fs.readFileSync(rawPath);
          resolve({ meta, data: rawData.buffer.slice(rawData.byteOffset, rawData.byteOffset + rawData.byteLength) });
        } else {
          reject(new Error('NIfTI读取失败'));
        }
      });
    });
  });

  // 读取raw体积数据
  ipcMain.handle('read-raw-volume', async (event, dirPath) => {
    const metaPath = path.join(dirPath, 'meta.json');
    const rawPath = path.join(dirPath, 'volume.raw');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    const rawData = fs.readFileSync(rawPath);
    return {
      meta: meta,
      data: rawData.buffer.slice(rawData.byteOffset, rawData.byteOffset + rawData.byteLength)
    };
  });

  // 读取配置文件
  ipcMain.handle('read-config', async (event, dirPath) => {
    try {
      // 先在当前目录查找
      let cfgPath = path.join(dirPath, 'mesh.cfg');
      console.log('查找配置文件:', cfgPath);
      
      // 如果当前目录没有，尝试在父目录查找
      if (!fs.existsSync(cfgPath)) {
        const parentDir = path.dirname(dirPath);
        cfgPath = path.join(parentDir, 'mesh.cfg');
        console.log('在父目录查找:', cfgPath);
      }
      
      // 如果还是没有，尝试在savestl子目录查找
      if (!fs.existsSync(cfgPath)) {
        cfgPath = path.join(dirPath, 'savestl', 'mesh.cfg');
        console.log('在savestl子目录查找:', cfgPath);
      }
      
      if (fs.existsSync(cfgPath)) {
        const content = fs.readFileSync(cfgPath, 'utf-8');
        console.log('配置文件内容:', content);
        const lines = content.split(/\r?\n/);
        const config = {};
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('[') || trimmed === '') continue;
          const eqIndex = trimmed.indexOf('=');
          if (eqIndex > 0) {
            const key = trimmed.substring(0, eqIndex).trim();
            const value = trimmed.substring(eqIndex + 1).trim();
            config[key] = value;
          }
        }
        console.log('解析后的配置:', config);
        return config;
      }
      console.log('配置文件不存在');
      return null;
    } catch (error) {
      console.error('读取配置文件失败:', error);
      return null;
    }
  });

  ipcMain.handle('return-path',async(event,args,modelPath)=>{
    try {
      const result = await handleresult(args, modelPath);
      console.log(result);
      return result;
    } catch (error) {
      console.error(error);
      return null; // 或者返回其他适当的错误信息
    }
  })

  // 调整窗口宽度
  ipcMain.handle('resize-window', async (event, widthDelta) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) {
      const [w, h] = win.getSize();
      win.setSize(w + widthDelta, h);
    }
  });

  // 检查是否有已有STL文件
  ipcMain.handle('check-existing-stl', async (event, outputFolder) => {
    const scapulaPath = path.join(outputFolder, 'Scapula.stl');
    const humerusPath = path.join(outputFolder, 'Humerus.stl');
    if (fs.existsSync(scapulaPath) && fs.existsSync(humerusPath)) {
      return { exists: true, scapulaPath, humerusPath };
    }
    return { exists: false };
  });

  // 检查是否有已有分割结果（在 inputFolder 中查找）
  ipcMain.handle('check-existing-segmentation', async (event, inputFolder) => {
    if (!fs.existsSync(inputFolder)) return { exists: false };
    const files = fs.readdirSync(inputFolder).filter(f => f.endsWith('.nii.gz') && !f.includes('_0000'));
    if (files.length === 0) return { exists: false };
    return { exists: true, path: path.join(inputFolder, files[0]) };
  });

  // nnU-Net推理
  ipcMain.handle('run-nnunet-inference', async (event, inputFolder, outputFolder) => {
    return new Promise((resolve, reject) => {
      const win = BrowserWindow.fromWebContents(event.sender);
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './run_inference.py', inputFolder, outputFolder
      ], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      let stdoutBuf = Buffer.alloc(0);
      let stderrBuf = Buffer.alloc(0);

      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        const text = stdoutBuf.toString();
        const lines = text.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line.trim());
              if (win && !win.isDestroyed()) {
                win.webContents.send('inference-progress', progress);
              }
            } catch (e) {
              console.log('[inference stdout]', line.trim());
            }
          }
        }
        stdoutBuf = Buffer.alloc(0);
      });

      pythonProcess.stderr.on('data', (data) => {
        stderrBuf = Buffer.concat([stderrBuf, data]);
        console.log('[inference stderr]', data.toString().trim());
      });

      pythonProcess.on('close', (code) => {
        console.log('推理进程退出, code:', code);
        if (code === 0) {
          resolve({ success: true, outputFolder: outputFolder });
        } else {
          reject(new Error(`推理进程退出 code ${code}: ${stderrBuf.toString().trim()}`));
        }
      });
    });
  });

  // NIfTI转STL
  ipcMain.handle('convert-nifti-to-stl', async (event, niftiPath, stlPath, labelId) => {
    return new Promise((resolve, reject) => {
      const win = BrowserWindow.fromWebContents(event.sender);
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './nifti_to_stl.py', niftiPath, stlPath, String(labelId)
      ], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      let stdoutBuf = Buffer.alloc(0);

      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        const text = stdoutBuf.toString();
        const lines = text.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line.trim());
              if (win && !win.isDestroyed()) {
                win.webContents.send('conversion-progress', progress);
              }
            } catch (e) {
              console.log('[convert stdout]', line.trim());
            }
          }
        }
        stdoutBuf = Buffer.alloc(0);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.log('[convert stderr]', data.toString().trim());
      });

      pythonProcess.on('close', (code) => {
        console.log('转换进程退出, code:', code);
        if (code === 0) {
          resolve({ success: true, stlPath: stlPath });
        } else {
          reject(new Error(`转换失败，退出码 ${code}`));
        }
      });
    });
  });

  // 按路径读取STL文件
  ipcMain.handle('read-stl-file', async (event, filePath) => {
    try {
      const data = fs.readFileSync(filePath);
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    } catch (error) {
      console.error('读取STL文件失败:', error);
      throw error;
    }
  });

  // 参数调整（计算新角度/长度）
  ipcMain.handle('adjust-parameters', async (event, params) => {
    try {
      const { center, dirPath } = params;
      const meshPath = path.join(dirPath, 'mesh.stl');
      const nailPath = path.join(dirPath, 'nail1.stl');
      if (!fs.existsSync(meshPath) || !fs.existsSync(nailPath)) {
        throw new Error('模型文件不存在');
      }
      // 调用 python 计算新参数
      const result = await new Promise((resolve, reject) => {
        const args = [
          './MasterWuVtkStlMaker.py',
          '--adjust',
          meshPath,
          nailPath,
          JSON.stringify(center)
        ];
        const proc = spawn('D:/Miniconda/envs/26guosai/python.exe', args, { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
        let buf = '';
        proc.stdout.on('data', d => buf += d.toString());
        proc.stderr.on('data', d => console.error('[adjust]', d.toString()));
        proc.on('close', code => {
          if (code !== 0) reject(new Error('Python退出码:' + code));
          else {
            try { resolve(JSON.parse(buf.trim())); }
            catch(e) { reject(new Error('解析结果失败')); }
          }
        });
      });
      return result;
    } catch(e) { console.error('[adjust-parameters]', e); return null; }
  });

  // 保存并重新生成模型
  ipcMain.handle('regenerate-models', async (event, params) => {
    try {
      const { dirPath, transformMatrix, length, angle1, angle2 } = params;
      console.log('[regenerate-models] 收到参数:', params);

      // 查找 mesh.cfg 文件（可能在多个位置）
      let cfgPath = path.join(dirPath, 'mesh.cfg');
      if (!fs.existsSync(cfgPath)) {
        // 尝试父目录
        cfgPath = path.join(path.dirname(dirPath), 'mesh.cfg');
        console.log('[regenerate-models] 在父目录查找:', cfgPath);
      }
      if (!fs.existsSync(cfgPath)) {
        // 尝试 savestl 子目录
        cfgPath = path.join(dirPath, 'savestl', 'mesh.cfg');
        console.log('[regenerate-models] 在savestl子目录查找:', cfgPath);
      }
      if (!fs.existsSync(cfgPath)) {
        // 尝试 savestl 下以患者ID命名的子目录
        const patientId = path.basename(dirPath);
        cfgPath = path.join(__dirname, 'savestl', patientId, 'mesh.cfg');
        console.log('[regenerate-models] 在savestl/患者ID目录查找:', cfgPath);
      }

      // 检查 mesh.cfg 是否存在
      if (!fs.existsSync(cfgPath)) {
        console.error('[regenerate-models] mesh.cfg 不存在');
        return {
          success: false,
          error: '请先完成选点流程（在3D模型上选取3个点）后再进行参数调整'
        };
      }

      console.log('[regenerate-models] 找到 cfg 文件:', cfgPath);
      const cfgDir = path.dirname(cfgPath);
      // 从 cfgDir 或 dirPath 中提取患者 ID
      let patientId = path.basename(cfgDir);
      console.log('[regenerate-models] cfgDir:', cfgDir, 'patientId:', patientId);
      console.log('[regenerate-models] __dirname:', __dirname);
      console.log('[regenerate-models] dirPath:', dirPath);

      // 患者目录（mesh.stl 的原始位置，由分割转换生成）
      const patientDir = path.join(__dirname, patientId);
      console.log('[regenerate-models] 患者目录:', patientDir);
      console.log('[regenerate-models] 患者目录 mesh.stl 存在:', fs.existsSync(path.join(patientDir, 'mesh.stl')));

      // 更新 cfg 的 new_parameter
      if (fs.existsSync(cfgPath)) {
        let cfgContent = fs.readFileSync(cfgPath, 'utf-8');
        if (!cfgContent.includes('[new_parameter]')) {
          cfgContent += '\n[new_parameter]\n';
        }
        // 更新或添加 new_parameter 值
        const updateCfg = (content, section, key, val) => {
          const regex = new RegExp(`(\\[${section}\\][^\\[]*?)(${key}\\s*=\\s*)([^\\n]*)`, 'g');
          if (regex.test(content)) {
            return content.replace(regex, `$1$2${val}`);
          } else {
            return content.replace(`[${section}]`, `[${section}]\n${key}=${val}`);
          }
        };
        cfgContent = updateCfg(cfgContent, 'new_parameter', 'location1', length);
        cfgContent = updateCfg(cfgContent, 'new_parameter', 'location2', angle1);
        cfgContent = updateCfg(cfgContent, 'new_parameter', 'location3', angle2);
        fs.writeFileSync(cfgPath, cfgContent, 'utf-8');
        console.log('[regenerate-models] cfg 已更新');
      } else {
        console.warn('[regenerate-models] cfg 文件不存在:', cfgPath);
      }
      // 从 cfg 读取原始旋转参数（如果前端未提供）
      let rawAngle1 = params.raw_angle1;
      let rawAngle2 = params.raw_angle2;
      if (rawAngle1 === undefined || rawAngle2 === undefined) {
        try {
          const cfgText = fs.readFileSync(cfgPath, 'utf-8');
          const m1 = cfgText.match(/raw_angle1\s*=\s*([-\d.eE]+)/);
          const m2 = cfgText.match(/raw_angle2\s*=\s*([-\d.eE]+)/);
          if (m1) rawAngle1 = parseFloat(m1[1]);
          if (m2) rawAngle2 = parseFloat(m2[1]);
          console.log('[regenerate-models] 从cfg读取原始旋转参数:', rawAngle1, rawAngle2);
        } catch(e) {}
      }
      // 调用 python 重新生成
      const result = await new Promise((resolve, reject) => {
        const args = [
          './MasterWuVtkStlMaker.py',
          '--regenerate',
          cfgDir,  // cfg 所在目录（savestl/0012）
          JSON.stringify({ length, angle1, angle2, raw_angle1: rawAngle1, raw_angle2: rawAngle2, transformMatrix }),
          patientDir  // mesh.stl 所在目录（0012）
        ];
        console.log('[regenerate-models] 启动 Python:', args);
        const proc = spawn('D:/Miniconda/envs/26guosai/python.exe', args, { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
        let stdoutBuf = Buffer.alloc(0);
        let stderrBuf = Buffer.alloc(0);
        proc.stdout.on('data', d => {
          stdoutBuf = Buffer.concat([stdoutBuf, d]);
          // 转发进度
          const text = d.toString();
          try {
            const obj = JSON.parse(text.trim());
            if (obj.progress !== undefined && mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('generate-progress', obj);
            }
          } catch(e) {}
        });
        proc.stderr.on('data', d => {
          stderrBuf = Buffer.concat([stderrBuf, d]);
          console.error('[regenerate]', d.toString());
        });
        proc.on('close', code => {
          console.log('[regenerate-models] Python退出码:', code);
          console.log('[regenerate-models] stdout:', stdoutBuf.toString());
          console.log('[regenerate-models] stderr:', stderrBuf.toString());
          if (code !== 0) {
            // 尝试从 stdout 中提取错误信息（我们的 emit 函数输出的）
            let errorMsg = '';
            const stdoutText = stdoutBuf.toString();
            const lines = stdoutText.split('\n');
            for (const line of lines) {
              try {
                const obj = JSON.parse(line.trim());
                if (obj.message && obj.message.includes('错误')) {
                  errorMsg = obj.message;
                  break;
                }
              } catch(e) {}
            }
            // 如果没有从 stdout 找到错误信息，使用 stderr
            if (!errorMsg) {
              errorMsg = stderrBuf.toString().trim().slice(-500);
            }
            reject(new Error('Python退出码:' + code + (errorMsg ? '\n' + errorMsg : '')));
          } else {
            // 重新读取 cfg 获取 Python 更新后的原始旋转参数
            try {
              const cfgText2 = fs.readFileSync(cfgPath, 'utf-8');
              const m1 = cfgText2.match(/raw_angle1\s*=\s*([-\d.eE]+)/);
              const m2 = cfgText2.match(/raw_angle2\s*=\s*([-\d.eE]+)/);
              if (m1) rawAngle1 = parseFloat(m1[1]);
              if (m2) rawAngle2 = parseFloat(m2[1]);
              console.log('[regenerate-models] Python更新后的原始旋转参数:', rawAngle1, rawAngle2);
            } catch(e) {}
            resolve({ success: true, dirPath: cfgDir, raw_angle1: rawAngle1, raw_angle2: rawAngle2 });
          }
        });
      });
      return result;
    } catch(e) { console.error('[regenerate-models]', e); return { success: false, error: e.message }; }
  });

  ipcMain.handle('toggle-dev-tools',async(event)=>{
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) {
      win.webContents.toggleDevTools();
    }
  })

  // AI 窗口控制
  ipcMain.handle('ai-minimize', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) win.minimize();
    return true;
  })
  ipcMain.handle('ai-maximize', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) { win.isMaximized() ? win.unmaximize() : win.maximize(); }
    return true;
  })
  ipcMain.handle('ai-close', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) win.close();
    return true;
  })

  // 打开 AI 助手窗口
  ipcMain.handle('open-ai-panel', async () => {
    if (aiWindow && !aiWindow.isDestroyed()) {
      aiWindow.focus();
      return;
    }
    aiWindow = new BrowserWindow({
      width: 420,
      height: 600,
      title: '术前规划智能分析助手',
      frame: false,
      resizable: true,
      minimizable: true,
      maximizable: true,
      webPreferences: {
        preload: path.join(__dirname, 'preload.js')
      }
    });
    aiWindow.loadFile('src/dading/ai_panel.html');
    aiWindow.on('closed', () => { aiWindow = null; });
    aiWindow.webContents.on('before-input-event', (event, input) => {
      if (input.key === 'F12') { aiWindow.webContents.toggleDevTools(); event.preventDefault(); }
    });
  });

  // 获取主窗口的 AI 上下文参数
  ipcMain.handle('get-ai-context', async () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      try {
        const ctx = await mainWindow.webContents.executeJavaScript(
          'window.getAiContextForExternal && window.getAiContextForExternal()'
        );
        console.log('[get-ai-context] ctx:', ctx);
        return ctx || {};
      } catch (e) { console.error('[get-ai-context] error:', e); return {}; }
    }
    console.log('[get-ai-context] mainWindow not available');
    return {};
  });

  // AI 助手对话（流式）
  ipcMain.handle('ai-chat', async (event, argJson) => {
    return new Promise((resolve, reject) => {
      // 将参数写入临时文件，避免 base64 图片导致命令行过长（ENAMETOOLONG）
      const tmpFile = path.join(os.tmpdir(), `ai_chat_${Date.now()}.json`);
      fs.writeFileSync(tmpFile, argJson, 'utf-8');
      console.log('[ai-chat] 启动, tmpFile:', tmpFile);

      // 流式数据发给 AI 窗口
      const targetWin = (aiWindow && !aiWindow.isDestroyed()) ? aiWindow : BrowserWindow.fromWebContents(event.sender);
      console.log('[ai-chat] targetWin:', targetWin ? 'exists' : 'null');
      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './ai_chat.py', '--file', tmpFile
      ], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      let stdoutBuf = Buffer.alloc(0);
      let stderrBuf = Buffer.alloc(0);

      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
        const text = stdoutBuf.toString();
        const lines = text.split('\n');
        // 最后一个元素可能是不完整的行，保留到下次处理
        stdoutBuf = Buffer.from(lines.pop() || '', 'utf-8');
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const msg = JSON.parse(trimmed);
            console.log('[ai-chat] stdout msg:', msg.type);
            if (targetWin && !targetWin.isDestroyed()) {
              targetWin.webContents.send('ai-chat-stream', msg);
            }
          } catch (e) {}
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        stderrBuf = Buffer.concat([stderrBuf, data]);
        console.error('[ai_chat stderr]', data.toString().trim());
      });

      pythonProcess.on('close', (code) => {
        console.log('[ai_chat] Python退出 code:', code);
        try { fs.unlinkSync(tmpFile); } catch (e) {}
        const remaining = stdoutBuf.toString().trim();
        if (remaining) {
          try {
            const msg = JSON.parse(remaining);
            if (targetWin && !targetWin.isDestroyed()) {
              targetWin.webContents.send('ai-chat-stream', msg);
            }
          } catch (e) {}
        }
        if (code === 0) {
          resolve({ success: true });
        } else {
          const errMsg = stderrBuf.toString().trim().slice(-500);
          console.error('[ai_chat] 错误:', errMsg);
          // 发送错误到 AI 窗口
          if (targetWin && !targetWin.isDestroyed()) {
            targetWin.webContents.send('ai-chat-stream', { type: 'error', message: `AI进程退出 code ${code}: ${errMsg}` });
          }
          resolve({ success: false, error: errMsg });
        }
      });
    });
  })

  // 保存 AI 反馈记录
  console.log('[main] 注册 save-ai-feedback handler');
  ipcMain.handle('save-ai-feedback', async (event, feedbackData) => {
    return new Promise((resolve, reject) => {
      const tmpFile = path.join(os.tmpdir(), `ai_feedback_${Date.now()}.json`);
      fs.writeFileSync(tmpFile, JSON.stringify(feedbackData), 'utf-8');
      console.log('[save-ai-feedback] 启动, tmpFile:', tmpFile);

      const pythonProcess = spawn('D:/Miniconda/envs/26guosai/python.exe', [
        './save_feedback.py'
      ], {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
        stdio: ['pipe', 'pipe', 'pipe']
      });

      // 通过 stdin 传入数据
      const input = fs.readFileSync(tmpFile, 'utf-8');
      pythonProcess.stdin.write(input);
      pythonProcess.stdin.end();

      let stdoutBuf = Buffer.alloc(0);
      pythonProcess.stdout.on('data', (data) => {
        stdoutBuf = Buffer.concat([stdoutBuf, data]);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error('[save-feedback stderr]', data.toString().trim());
      });

      pythonProcess.on('close', (code) => {
        try { fs.unlinkSync(tmpFile); } catch (e) {}
        console.log('[save-ai-feedback] Python退出 code:', code);
        const output = stdoutBuf.toString().trim();
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          resolve({ success: code === 0, error: output });
        }
      });
    });
  });

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})