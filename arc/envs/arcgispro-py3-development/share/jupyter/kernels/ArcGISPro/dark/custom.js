(function() {
  init();

  function init() {
    disableKeyboardShortcuts();
    setupEventListeners();
    sendNotebookServerOSRequestMessage();
  }

  function disableKeyboardShortcuts()
  {
    IPython.keyboard_manager.command_shortcuts.remove_shortcut('0,0');
  }

  function receiveMessage(event) {
    if ((event.data || {}).messageType === "NotebookMessage") {
      insertItemCell(event);
    }

    if ((event.data || {}).messageType === "NotebookSaveMessage") {
      Jupyter.notebook.save_notebook();
    }

    if ((event.data || {}).messageType === "NotebookSaveAsMessage") {
      sendNotebookJson();
    }

    if ((event.data || {}).messageType === "NotebookServerOSResponseMessage") {
      createHelpMenus(event.data.message.serverOS);
    }
  }

  function insertItemCell(event) {
    Jupyter.notebook.insert_cell_below();
    Jupyter.notebook.select_next();
    var cell = Jupyter.notebook.get_selected_cell();

    if (cell) {
      cell.code_mirror.setValue(event.data.message);
    } else {
      console.warn("Unable to interact with notebook because no cells are currently selected.");
    }
  }

  function sendNotebookServerOSRequestMessage() {
    window.parent.postMessage(
      {
        messageType: "NotebookServerOSRequestMessage"
      },
      "*"
    );
  }

  function sendDirtyStatus(event, data) {
    window.parent.postMessage(
      {
        messageType: "DirtyStatusMessage",
        message: {
          dirty: data.value
        }
      },
      "*"
    );
  }

  function sendCheckpointStatus(event, data) {
    sendSavedStatus();
    window.parent.postMessage(
      {
        messageType: "CheckpointStatusMessage",
        message: {
          checkpoint: data
        }
      },
      "*"
    );
  }

  function sendNotebookReadyStatus() {
    window.parent.postMessage(
      {
        messageType: "InitialNotebookStatus",
        message: {
          dirty: Jupyter.notebook.dirty,
          checkpoints: Jupyter.notebook.checkpoints
        }
      },
      "*"
    );
  }

  function sendSavedStatus(event) {
    window.parent.postMessage(
      {
        messageType: "SavedStatusMessage",
        message: {
          saved: true
        }
      },
      "*"
    );
  }

  function sendKernelKilledStatus(event) {
    Jupyter.notification_area.widget("kernel").warning("kernel disconnected! Reopen notebook");
  }

  function setupEventListeners() {
    window.addEventListener("message", receiveMessage, false);
    Jupyter.notebook.events.on("set_dirty.Notebook", sendDirtyStatus);
    Jupyter.notebook.events.on("checkpoint_created.Notebook", sendCheckpointStatus);
    Jupyter.notebook.events.on("notebook_saved.Notebook", sendSavedStatus);
    Jupyter.notebook.events.on("kernel_ready.Kernel", sendNotebookReadyStatus);
    Jupyter.notebook.events.on('kernel_killed.Kernel kernel_killed.Session', sendKernelKilledStatus);
  }

  function createHelpMenus(serverOS) {
    var context = getContext();
    var startNode = document.getElementById("edit_keyboard_shortcuts");
    var divider = document.createElement("li");
    divider.setAttribute("class", "divider");

    var hyphenatedLocales = ["zh-CN", "pt-BR"];
    var locale = hyphenatedLocales.indexOf(navigator.language) === -1 ? navigator.language.split("-")[0] : navigator.language;

    var userGuideUrl = "/" + context + "nbhelp/admin/" + locale + "/notebook";
    var pythonApiHelpUrl = "/" + context + "nbhelp/" + locale + "/python";
    var arcPyHelpUrl = userGuideUrl + "/latest/python/" + serverOS + "/use-arcpy-in-your-notebook.htm";

    var notebookUserGuide = createMenuItem("notebook_user_guide", "ArcGIS Notebook Server User Guide", userGuideUrl);
    var pythonApiHelp = createMenuItem("python_api_help", "ArcGIS API for Python Help", pythonApiHelpUrl);
    var arcPyHelp = createMenuItem("arcpy_help", "ArcPy Help", arcPyHelpUrl);

    startNode.insertAdjacentElement("afterend", divider);
    divider.insertAdjacentElement("afterend", notebookUserGuide);
    notebookUserGuide.insertAdjacentElement("afterend", pythonApiHelp);

    if(serverOS){
      pythonApiHelp.insertAdjacentElement("afterend", arcPyHelp);
    }
  }

  function createMenuItem(id, text, url) {
    var menuItem = document.createElement("li");
    var menuItemLink = document.createElement("a");
    var menuItemIcon = document.createElement("i");

    menuItem.setAttribute("id", id);
    menuItemLink.setAttribute("href", url);
    menuItemLink.setAttribute("target", "_blank");
    menuItemIcon.setAttribute("class", "fa fa-external-link menu-icon pull-right");

    menuItemLink.appendChild(document.createTextNode(text));
    menuItemLink.appendChild(menuItemIcon);
    menuItem.appendChild(menuItemLink);

    return menuItem;
  }

  function sendNotebookJson() {
    // this is to prevent unsaved changes prompt
    // when saving a shared notebook
    window.onbeforeunload = null;

    var notebookJson = Jupyter.notebook.toJSON();

    window.parent.postMessage(
      {
        messageType: "NotebookJsonMessage",
        message: {
          notebookJson: notebookJson
        }
      },
      "*"
    );
  }

  function getContext() {
    var pathParts = window.location.pathname.split("/").filter(function(part) { return !!part; });

    return pathParts.length ? pathParts[0] + "/" : "";
  }
})();
