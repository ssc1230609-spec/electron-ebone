const { contextBridge,ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('electronapi',{
    path:(path)=>ipcRenderer.invoke('return-path',path)
})