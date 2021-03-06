package com.example.mltest;

import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.Kwarg;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import android.annotation.TargetApi;
import android.content.Intent;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.graphics.Bitmap;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.net.URI;

public class MainActivity extends AppCompatActivity {
    private Python py;
    private Button mButton;
    private EditText mEdit;
    private EditText mEditConv;
    private EditText mEditDense;
    private TextView mRes;
    private ProgressBar mTrainPercent;
    private Button mGetFilePath;
    private TextView mPrediction;
    private ProgressBar mLoading;
    private Button mDrawTest;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Python.start(new AndroidPlatform(this));
        py=Python.getInstance();
        mRes=(TextView)findViewById(R.id.res);
        mTrainPercent=(ProgressBar)findViewById(R.id.trainPercent);
//        py.getModule("main").callAttr("main");
        mButton=(Button)findViewById(R.id.button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                mButton.setText("Training Started");
//                PyObject y=py.getModule("main");
                String seconds= mEdit.getText().toString();
                String convs=mEditConv.getText().toString();
                String dense=mEditDense.getText().toString();
                int kw=Integer.parseInt(seconds);
                new trainTask().execute(kw,Integer.parseInt(convs),Integer.parseInt(dense));
//
//                PyObject x=y.callAttr("main",new Kwarg("second",kw));
//                mButton.setText(x.toString());
            }
        });
        mEdit=(EditText)findViewById(R.id.edit);
        mEditConv=(EditText)findViewById(R.id.editText);
        mEditDense=(EditText)findViewById(R.id.editText2);
        mGetFilePath=(Button)findViewById(R.id.getFilePath);
        mPrediction=(TextView) findViewById(R.id.prediction);
        mLoading=(ProgressBar)findViewById(R.id.progressBar);
        mLoading.setVisibility(View.GONE);

            mGetFilePath.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent,0);
            }
        });
            mDrawTest = (Button)findViewById(R.id.getDraw);
            mDrawTest.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    /*




                     */
                    Intent i = new Intent(MainActivity.this,DrawActivity.class);
                    startActivityForResult(i,143);

                }
            });


    }
    private class testTask extends AsyncTask<Uri, Void, Integer>{
        @Override
        protected void onPreExecute(){
            super.onPreExecute();
            mGetFilePath.setText("Testing Started");
            mLoading.setVisibility(View.VISIBLE);

        }

        @Override
        protected Integer doInBackground(Uri... uris) {
            Bitmap bitmap = null;
            Uri target = uris[0];
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), target);
            }
            catch( Exception e){

            }
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 75, stream);
            byte[] byteArray = stream.toByteArray();
            bitmap.recycle();
            PyObject x=py.getModule("main");
            PyObject   y=x.callAttr("run",byteArray);
            return Integer.parseInt(y.toString());

        }

        @Override
        protected void onPostExecute(Integer integer) {
            super.onPostExecute(integer);
            mPrediction.setText("Prediction: "+integer.toString());

            mLoading.setVisibility(View.GONE);
        }
    }
    private class trainTask extends AsyncTask<Integer,Void,Double>{
        @Override
        protected void onPreExecute(){
            super.onPreExecute();
            mButton.setText("Training Started");
            mLoading.setVisibility(View.VISIBLE);


        }
        @Override
        protected Double doInBackground(Integer... params){
                int time=params[0];
                int convols=params[1];
                int denses=params[2];
//                PyObject y=py.getModule("main");



                PyObject x=py.getModule("main");
                PyObject   y=x.callAttr("main",new Kwarg("second",time), new Kwarg("conv",convols), new Kwarg("dens",denses));

                Double acc=Double.parseDouble(y.toString())*100.0;
                //x.callAttr("run","");
                x.close();
                y.close();
                return acc;

        }
        @Override
        @TargetApi(24)
        protected void onPostExecute(Double acc){
            super.onPostExecute(acc);
            String fin=acc.toString();
            if (fin.length()>5){
                fin=fin.substring(0,6);
            }
            mRes.setText(fin+"%");

            mTrainPercent.setProgress((int)(Double.parseDouble(fin)));
            mButton.setText("Train Again");
            mLoading.setVisibility(View.GONE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
            if(requestCode==0) {
                Uri target = data.getData();
                new testTask().execute(target);
            }
            else{
                Uri target = Uri.parse("file://"+Environment.getExternalStorageDirectory()+ File.separator +"drawing.JPEG");
                new testTask().execute(target);
            }


            //PyObject x=py.getModule("main");
            //PyObject shape= x.callAttr("test",target.getPath().toString());
            //mPrediction.setText(shape.toString());

    }
}
