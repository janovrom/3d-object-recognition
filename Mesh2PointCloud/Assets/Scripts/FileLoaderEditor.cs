using System;
using System.IO;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(FileLoader))]
public class FileLoaderEditor : Editor {


    private static GUIContent[] _dropdownContent;
    private int _selectedItem = 0;


    static FileLoaderEditor()
    {
        _dropdownContent = new GUIContent[0];
    }

    private void HandleDropdown()
    {
        _selectedItem = EditorGUILayout.Popup(new GUIContent("Selected Object:"), _selectedItem, _dropdownContent);
    }

    private void HandleReload()
    {
        GUIContent content = new GUIContent();
        content.text = "Reload";
        content.tooltip = "Reloads files in directory.";
        if (GUILayout.Button(content))
        {
            string path = Path.Combine(Directory.GetCurrentDirectory(), "data");
            if (Directory.Exists(path))
            {
                var files = Directory.GetFiles(path);
                _dropdownContent = new GUIContent[files.Length];
                for (int i = 0; i < files.Length; ++i)
                {
                    _dropdownContent[i] = new GUIContent();
                    _dropdownContent[i].text = Path.GetFileName(files[i]);
                    _dropdownContent[i].tooltip = files[i];
                }
                
            }
            else
            {
                Debug.LogWarning(string.Format("Directory {0} not found!", path));
            }
        }
    }

    private void HandleAxisTick()
    {
        GUIContent content = new GUIContent();
        content.text = "Switch Z/Y axes";
        content.tooltip = "Unity might use different axes than your file format. And it's probably switched Z,Y.";
        ((FileLoader)target).SwitchZY = GUILayout.Toggle(((FileLoader)target).SwitchZY, content);
    }

    private void HandleLoading()
    {
        GUIContent content = new GUIContent();
        content.text = "Load .off";
        content.tooltip = "Loads selected .off file.";
        //EditorGUILayout.DropdownButton(content, FocusType.Keyboard);
        if (GUILayout.Button(content))
        {
            if (_dropdownContent.Length > 0)
            {
                var filepath = _dropdownContent[_selectedItem].tooltip;
                ((FileLoader)target).LoadMesh(filepath);
            }
        }
    }

    public override void OnInspectorGUI()
    {
        HandleDropdown();
        HandleReload();
        HandleAxisTick();
        HandleLoading();
    }
}
#endif