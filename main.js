const { app, BrowserWindow ,ipcMain} = require('electron/main')
const { spawn } = require('child_process');
const path = require('node:path')
try {
    require('electron-reloader')(module)
  } catch (_) {}
function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
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
}
async function handleresult(args){
  return new Promise(async (resolve,reject)=>{
    const fpath = args;
    // 将二维数组转换为JSON字符串
    const jsonFpath = JSON.stringify(fpath);

    try{
      const pythonProcess = await spawn('python', ['./zidongnail/MasterWuVtkStlMaker.py', jsonFpath]);
      let number = ''
      pythonProcess.stdout.on('data', (data) => {
        number += data.toString().trim();
        console.log(number)
      });
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          console.log(number)
          resolve(number); // 成功时，将结果传递给Promise的resolve函数
        } else {
          reject(new Error(`Python process exited with code ${code}`)); // 失败时，将错误传递给Promise的reject函数
        }
      });
    } catch (error) {
      reject(error); // 捕获到异常时，将错误传递给Promise的reject函数
    }
  })
}
app.whenReady().then(() => {
  ipcMain.handle('return-path',async(event,args)=>{
    try {
      const result = await handleresult(args);
      console.log(result);
      return result;
    } catch (error) {
      console.error(error);
      return null; // 或者返回其他适当的错误信息
    }
  })
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