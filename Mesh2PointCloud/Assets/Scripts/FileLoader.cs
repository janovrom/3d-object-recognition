using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[ExecuteInEditMode]
public class FileLoader : MonoBehaviour {

    public bool SwitchZY = true;
    private string _outputDirectory;

	// Use this for initialization
	void Start () {
        _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "data-out");
    }
	
	// Update is called once per frame
	void Update () {
		
	}

    public void SavePoints(Color[] positions, int Width, int Height, string name)
    {
        // Convert texture to point cloud
        using (StreamWriter writer = new StreamWriter(File.Create(Path.Combine(_outputDirectory, name))))
        {
            for (int i = 0; i < Width; ++i)
            {
                for (int j = 0; j < Height; ++j)
                {
                    Color c = positions[j * Width + i];
                    float a = c.a;
                    float r = c.r;
                    float g = c.g;
                    float b = c.b;

                    if (Mathf.Abs(r) < 0.001 && Mathf.Abs(g) < 0.001 && Mathf.Abs(b) < 0.001)
                    {
                        continue;
                    }

                    writer.Write(r);
                    writer.Write(" ");
                    writer.Write(g);
                    writer.Write(" ");
                    writer.Write(b);
                    writer.Write("\n");
                }
            }
        }
    }

    public GameObject LoadMesh(string path)
    {
        Debug.Log("Loading mesh from " + path);
        var lines = File.ReadAllLines(path);
        // Check if it is really off file format
        Debug.Assert(lines[0].Equals("OFF"));
        // Read the header
        var header = lines[1].Split(' ');
        int numVertices = 0;
        int numFaces = 0;
        int numEdges = 0;
        bool success = true;
        success &= int.TryParse(header[0], out numVertices);
        success &= int.TryParse(header[1], out numFaces);
        success &= int.TryParse(header[2], out numEdges);
        // Check header validity
        Debug.Assert(success, "Off file header is not valid!");
        int header_size = 2;
        Vector3[] vertices = new Vector3[numVertices];
        int[] indices = new int[numFaces * 3];

        // Read vertices
        for (int i = header_size; i < numVertices + header_size; ++i)
        {
            var splitted = lines[i].Split(' ');
            float x, y, z;
            if (SwitchZY)
            {
                success &= float.TryParse(splitted[0], out x);
                success &= float.TryParse(splitted[1], out z);
                success &= float.TryParse(splitted[2], out y);
            }
            else
            {
                success &= float.TryParse(splitted[0], out x);
                success &= float.TryParse(splitted[1], out y);
                success &= float.TryParse(splitted[2], out z);
            }
            Debug.Assert(success, "Couldn't parse " + (i - 2).ToString() + " vertex!");

            vertices[i - 2] = new Vector3(x, y, z);
        }

        // Read indices
        for (int i = numVertices + header_size; i < numVertices + header_size + numFaces; ++i)
        {
            int ii = i - numVertices - header_size;
            var splitted = lines[i].Split(' ');
            Debug.Assert(splitted.Length == 4, "Reading wrong line for face " + (ii).ToString() + " or mesh is not triangulated!");
            int cc;
            int idx1, idx2, idx3;
            // Read index count for this face
            success &= int.TryParse(splitted[0], out cc);
            Debug.Assert(success, "Something went wrong when parsing " + lines[i]);
            Debug.Assert(cc == 3, "Face " + (ii).ToString() + " is not a triangle.");

            // Read indices for this face
            success &= int.TryParse(splitted[1], out idx1);
            success &= int.TryParse(splitted[2], out idx2);
            success &= int.TryParse(splitted[3], out idx3);
            Debug.Assert(success, "Something went wrong when parsing " + lines[i]);

            indices[ii * 3 + 0] = idx1;
            indices[ii * 3 + 1] = idx2;
            indices[ii * 3 + 2] = idx3;

            // Compute normal
            Vector3 v2v1 = vertices[idx2] - vertices[idx1];
            Vector3 v3v1 = vertices[idx3] - vertices[idx1];
            Vector3 n = Vector3.Cross(v2v1, v3v1);
        }

        GameObject go = new GameObject(Path.GetFileNameWithoutExtension(path));
        go.AddComponent<MeshRenderer>();
        MeshFilter mf = go.AddComponent<MeshFilter>();

        if (mf.sharedMesh != null)
        {
            mf.sharedMesh.Clear();
        }
        else
        {
            mf.sharedMesh = new Mesh();
        }

        mf.sharedMesh.vertices = vertices;
        mf.sharedMesh.triangles = indices;
        mf.sharedMesh.RecalculateBounds();
        mf.sharedMesh.RecalculateNormals();
        return go;
    }
}
