"use strict";
const electron = require("electron");
const api = {
  openFolder: () => electron.ipcRenderer.invoke("dialog:openFolder"),
  openFile: (opts) => electron.ipcRenderer.invoke("dialog:openFile", opts),
  listImages: (dir) => electron.ipcRenderer.invoke("fs:listImages", dir),
  scanArBuckets: (opts) => electron.ipcRenderer.invoke("dataset:scanArBuckets", opts),
  readCaption: (imagePath) => electron.ipcRenderer.invoke("fs:readCaption", imagePath),
  writeCaption: (imagePath, text) => electron.ipcRenderer.invoke("fs:writeCaption", imagePath, text),
  deleteImage: (imagePath) => electron.ipcRenderer.invoke("fs:deleteImage", imagePath),
  readImageMeta: (imagePath) => electron.ipcRenderer.invoke("fs:readImageMeta", imagePath),
  readImageBase64: (imagePath) => electron.ipcRenderer.invoke("fs:readImageBase64", imagePath),
  getSettings: () => electron.ipcRenderer.invoke("settings:get"),
  setSettings: (settings) => electron.ipcRenderer.invoke("settings:set", settings),
  relaunchApp: () => electron.ipcRenderer.invoke("app:relaunch"),
  listGpuDevices: () => electron.ipcRenderer.invoke("gpu:listDevices"),
  getResourceStats: (deviceId) => electron.ipcRenderer.invoke("system:getResourceStats", deviceId),
  killProcess: (pid) => electron.ipcRenderer.invoke("system:killProcess", pid),
  defaultDownloadFolder: () => electron.ipcRenderer.invoke("download:defaultFolder"),
  probePython: (pythonPath) => electron.ipcRenderer.invoke("python:probe", pythonPath),
  installPython: (opts) => electron.ipcRenderer.invoke("python:install", opts),
  cancelPythonInstall: () => electron.ipcRenderer.invoke("python:cancelInstall"),
  onPythonInstallProgress: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("python:installProgress", listener);
    return () => electron.ipcRenderer.removeListener("python:installProgress", listener);
  },
  startTrain: (opts) => electron.ipcRenderer.invoke("train:start", opts),
  stopTrain: () => electron.ipcRenderer.invoke("train:stop"),
  trainStatus: () => electron.ipcRenderer.invoke("train:status"),
  listTrainCheckpoints: (opts) => electron.ipcRenderer.invoke("train:listCheckpoints", opts),
  listTrainSamples: (opts) => electron.ipcRenderer.invoke("train:listSamples", opts),
  onTrainLog: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("train:log", listener);
    return () => electron.ipcRenderer.removeListener("train:log", listener);
  },
  onTrainProgress: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("train:progress", listener);
    return () => electron.ipcRenderer.removeListener("train:progress", listener);
  },
  onTrainLossSpike: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("train:lossSpike", listener);
    return () => electron.ipcRenderer.removeListener("train:lossSpike", listener);
  },
  onTrainDone: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("train:done", listener);
    return () => electron.ipcRenderer.removeListener("train:done", listener);
  },
  onTrainError: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("train:error", listener);
    return () => electron.ipcRenderer.removeListener("train:error", listener);
  },
  checkModelStatus: (opts) => electron.ipcRenderer.invoke("model:checkStatus", opts),
  downloadModel: (opts) => electron.ipcRenderer.invoke("model:download", opts),
  cancelModelDownload: () => electron.ipcRenderer.invoke("model:cancelDownload"),
  onModelDownloadProgress: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("model:downloadProgress", listener);
    return () => electron.ipcRenderer.removeListener("model:downloadProgress", listener);
  },
  onModelDownloadDone: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("model:downloadDone", listener);
    return () => electron.ipcRenderer.removeListener("model:downloadDone", listener);
  },
  onModelDownloadError: (cb) => {
    const listener = (_e, payload) => cb(payload);
    electron.ipcRenderer.on("model:downloadError", listener);
    return () => electron.ipcRenderer.removeListener("model:downloadError", listener);
  },
  ensureWd14Model: (opts) => electron.ipcRenderer.invoke("wd14:ensureModel", opts),
  tagWd14: (opts) => electron.ipcRenderer.invoke("wd14:tag", opts),
  cancelWd14: () => electron.ipcRenderer.invoke("wd14:cancel"),
  toLocalUrl: (filePath) => {
    const normalized = filePath.replace(/\\/g, "/");
    const encoded = normalized.split("/").map((seg) => encodeURIComponent(seg)).join("/");
    return `local-file://local/${encoded}`;
  }
};
electron.contextBridge.exposeInMainWorld("api", api);
