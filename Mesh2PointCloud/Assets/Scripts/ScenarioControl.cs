﻿using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ScenarioControl : MonoBehaviour {

    public RenderTexture ScreenTexture;
    public Material CameraPositionMaterial;
    public RawImage GuiScreenTexture = null;
    private Texture2D _copyTexture;
    private FileLoader _fileLoader;
    private bool _savePoints = false;
    private int Width = 224;
    private int Height = 172;
    private float _tan = 0.0f;
    private string[] _paths;
    private int _currentObject = 0;
    // Start as false, so we can wait for user input
    private bool _finishedCurrentObject = false;
    // Scenario variables
    private float _dist = 0.0f;
    private GameObject _currentGameObject;
    // Rotation indices
    private int i = 0;
    private int j = 0;
    private int _dataIdx = 0;


    private void LoadObjectsNames()
    {
        string path = Path.Combine(Directory.GetCurrentDirectory(), "data");
        if (Directory.Exists(path))
        {
            var files = Directory.GetFiles(path);
            _paths = new string[files.Length];
            for (int i = 0; i < files.Length; ++i)
            {
                _paths[i] = files[i];
            }
        }
        else
        {
            Debug.LogWarning(string.Format("Directory {0} not found!", path));
        }
    }

	// Use this for initialization
	void Start () {
        Screen.SetResolution(Width, Height, false);
        // Create render texture as big as screen
        ScreenTexture = new RenderTexture(Width, Height, 16, RenderTextureFormat.ARGBFloat);
        ScreenTexture.Create();
        // Create one copy texture with the same size as the rendering one
        _copyTexture = new Texture2D(Width, Height, TextureFormat.RGBAFloat, false);
        // Attach it to the one and only main camera
        Camera.main.targetTexture = ScreenTexture;
        GuiScreenTexture.texture = ScreenTexture;
        // Get FileLoader for saving points
        _fileLoader = FindObjectOfType<FileLoader>();

        _tan = Mathf.Tan(Mathf.Deg2Rad * Camera.main.fieldOfView / 2.0f);

        LoadObjectsNames();
    }

    private void LoadObject()
    {
        if (_currentGameObject)
        {
            Destroy(_currentGameObject);
        }
        _currentGameObject = _fileLoader.LoadMesh(_paths[_currentObject]);
        // Center object to zero and hope it was centered in the data
        _currentGameObject.transform.position = Vector3.zero;
        _currentGameObject.GetComponent<MeshRenderer>().sharedMaterial = CameraPositionMaterial;
        Bounds bs = _currentGameObject.GetComponent<MeshFilter>().sharedMesh.bounds;
        // Get radius of bounding sphere as the largest extent
        float r = bs.extents.magnitude;
        // We have tangens and side "a", so compute minimum distance from object center, so it will be seen all
        _dist = r / _tan;
        // Move the camera
        Camera.main.transform.position = new Vector3(0.0f,1.0f,-1.0f).normalized * _dist;
        Camera.main.transform.LookAt(_currentGameObject.transform);
        _finishedCurrentObject = false;
    }

    void Update()
    {
        // We saved all objects as point clouds
        if (_currentObject >= _paths.Length)
            _savePoints = false;

        if (!_savePoints)
            return;


        // Select object and create correct rotations and display it
        if (_finishedCurrentObject)
        {
            LoadObject();
        }

        float rotY = j * 60.0f;
        // Apply transforms
        _currentGameObject.transform.localRotation = Quaternion.Euler(0.0f, rotY, 0.0f);

        // Increment state-rotation variables
        ++j;
        if (j == 6)
        {
            j = 0;
            ++i;
            _finishedCurrentObject = true;
        }
        //// Finished rotating the object
        //if (i == 4)
        //{
        //    i = 0;
        //    j = 0;
        //}
    }

	void OnPostRender() {

        if (!_savePoints)
        {
            return;
        }

        // Copy data to texture from which we can read and save points
        _copyTexture.ReadPixels(new Rect(0, 0, ScreenTexture.width, ScreenTexture.height), 0, 0, false);
        _copyTexture.Apply();
        Color[] data = _copyTexture.GetPixels();
        _fileLoader.SavePoints(data, Width, Height, Path.GetFileNameWithoutExtension(_paths[_currentObject]) + "_" + _dataIdx.ToString() + ".xyz");
        ++_dataIdx;
        // If we saved and used last rotation of the object, increment pointer
        if (_finishedCurrentObject)
        {
            ++_currentObject;
        }
    }

    public void SavePoints()
    {
        _savePoints = true;
        LoadObject();
    }

}
