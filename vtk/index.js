var btn = document.getElementById('icon');
btn.onclick=async function() {//实现打开文件资源管理器

  try {
     // 获得文件夹的句柄
     const handle = await showDirectoryPicker();
     console.log(handle)
     const root = await processHandler(handle);
   //   获得文件内容
    const file = await root.children[1].getFile();
  
    const reader = new FileReader();
    reader.onload=e=>{
      // 读取结果
      console.log(e.target.result)
    }
    reader.readAsText(file,'utf-8')
   }
   catch {
     //用户拒绝查看文件
    //  alert('访问失败')
   }
}
async function processHandler(handle) {
  if (handle.kind==='file'){
    return handle
  }
    handle.children=[]
    const iter = await handle.entries();//获得文件夹中的所有内容
    //iter:异步迭代器
    for await (const info of iter){
      var subHandle = await processHandler(info[1]);
      handle.children.push(subHandle)
    }
    return handle
}
