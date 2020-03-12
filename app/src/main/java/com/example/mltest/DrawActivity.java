package com.example.mltest;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.gesture.GestureOverlayView;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;

import java.io.File;
import java.io.FileOutputStream;

public class DrawActivity extends AppCompatActivity {
    private GestureOverlayView mDraw;
    private Button mSave;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        mDraw=(GestureOverlayView)findViewById(R.id.drawPad);
        mSave=(Button) findViewById(R.id.Save);
        mSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                saveDraw();
                finish();
            }
        });
    }
    public void saveDraw(){
        mDraw.setDrawingCacheEnabled(true);
        Bitmap map = Bitmap.createBitmap(mDraw.getDrawingCache());
        File ur = new File(Environment.getExternalStorageDirectory()+File.separator +"drawing.JPEG");
        try {
            ur.createNewFile();
            FileOutputStream out = new FileOutputStream(ur);
            map.compress(Bitmap.CompressFormat.JPEG, 75, out);
            out.close();
        }
        catch( Exception e){
            mSave.setText("Try Again");
        }

    }
}
