const APP_MODE = "dev";
let API_URL = "";
if(APP_MODE === "dev"){
    API_URL = "http://127.0.0.1:5000/api";
}else{
    API_URL = "prod";
}
localStorage.setItem("appUrl",API_URL);