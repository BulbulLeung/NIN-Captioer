"use strict";
const electron = require("electron");
const api = {
  openFolder: () => electron.ipcRenderer.invoke("dialog:openFolder"),
  listImages: (dir) => electron.ipcRenderer.invoke("fs:listImages", dir),
  readCaption: (imagePath) => electron.ipcRenderer.invoke("fs:readCaption", imagePath),
  writeCaption: (imagePath, text) => electron.ipcRenderer.invoke("fs:writeCaption", imagePath, text),
  deleteImage: (imagePath) => electron.ipcRenderer.invoke("fs:deleteImage", imagePath),
  readImageMeta: (imagePath) => electron.ipcRenderer.invoke("fs:readImageMeta", imagePath),
  readImageBase64: (imagePath) => electron.ipcRenderer.invoke("fs:readImageBase64", imagePath),
  getSettings: () => electron.ipcRenderer.invoke("settings:get"),
  setSettings: (settings) => electron.ipcRenderer.invoke("settings:set", settings),
  toLocalUrl: (filePath) => {
    const normalized = filePath.replace(/\\/g, "/");
    const encoded = normalized.split("/").map((seg) => encodeURIComponent(seg)).join("/");
    return `local-file://local/${encoded}`;
  }
};
electron.contextBridge.exposeInMainWorld("api", api);
